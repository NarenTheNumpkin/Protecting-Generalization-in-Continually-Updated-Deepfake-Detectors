"""J4 job queue (registration/journal_J4.md). Worker 0 first runs the inference script (blocks B, C); the development runs
of block A follow; the matched settings are chosen mechanically by the registered rule, then their test runs are queued."""

import paths
import json
import os
import subprocess
import sys
import threading
import time

R = paths.EXPERIMENTS
PY = paths.PYTHON
INIT = paths.INIT_CKPT
CHOICE = f"{R}/registration/J4_choice.json"
DEV = {"realkd_lwf": ("realkd_lwf", []),
       **{f"lwf_l{l}": ("lwf", ["--lam-lwf", l]) for l in ["1", "2", "4", "10"]},
       **{f"seq_lr{lr}": ("seq", ["--lr", lr]) for lr in ["3e-5", "1e-5"]},
       **{f"realkd_l{l}": ("realkd", ["--lam", l]) for l in ["1", "3", "10"]}}
FAMILY = {"lwf": (["lwf_l1", "lwf_l2", "lwf_l4", "lwf_l10"], "lwf_l1"),  # candidates in increasing regularisation, default
          "seq": (["j3:seq_lr1e-4", "seq_lr3e-5", "seq_lr1e-5"], "j3:seq_lr1e-4"),
          "realkd": (["realkd_l1", "realkd_l3", "realkd_l10"], "realkd_l1")}


def dev_path(name):
    return f"{R}/runs/j3_gpl_dev/{name[3:]}/results.json" if name.startswith("j3:") else f"{R}/runs/j4_dev/{name}/results.json"


def finished(p, dev):
    if not os.path.exists(p):
        return False
    r = json.load(open(p))
    return (len(r["stages"]) == len(r["args"]["stream"].split(","))) if dev else ("panel" in r)


def gain(p):
    r = json.load(open(p))
    st, stream = r["stages"], r["args"]["stream"].split(",")
    return 100 * sum(st[j]["dev"][stream[j]]["auroc"] - st[j - 1]["dev"][stream[j]]["auroc"] for j in range(1, len(stream))) / (len(stream) - 1)


JOBS = [{"name": "infer", "cmd": [PY, "j4_infer.py"], "log": f"{R}/analysis_j4_infer.log", "done": lambda: os.path.exists(f"{R}/analysis_j4/preds32_gpl_CDF.json")}]
for name, (m, extra) in DEV.items():
    JOBS.append({"name": f"j4_dev/{name}", "cmd": [PY, "stream.py", "--method", m, "--tag", "j4_dev", "--run-name", name, "--init", INIT] + extra,
                 "log": f"{R}/runs/j4_dev/{name}.log", "done": (lambda n=name: finished(dev_path(n), True))})
lock = threading.Lock()


def try_choice():
    if os.path.exists(CHOICE):
        return True
    names = ["realkd_lwf"] + [c for f in FAMILY.values() for c in f[0]]
    if not all(finished(dev_path(n), True) for n in names):
        return False
    target = gain(dev_path("realkd_lwf"))
    ch = {"target_gain": target, "dev_gain": {n: gain(dev_path(n)) for n in names}, "chosen": {}}
    for fam, (cands, default) in FAMILY.items():
        best = min(range(len(cands)), key=lambda i: (abs(ch["dev_gain"][cands[i]] - target), -i))  # ties: more regularised
        ch["chosen"][fam] = {"setting": cands[best], "reuse_existing": cands[best] == default}
    json.dump(ch, open(CHOICE, "w"), indent=1)
    print("J4 choice", json.dumps(ch), flush=True)
    for fam, c in ch["chosen"].items():
        if c["reuse_existing"]:
            continue
        name = c["setting"].replace("j3:", "")
        extra = DEV[name][1] if name in DEV else []
        for s in range(5):
            JOBS.append({"name": f"j4_matched/{name}_seed{s}", "log": f"{R}/runs/j4_matched/{name}_seed{s}.log",
                         "cmd": [PY, "stream.py", "--method", fam, "--tag", "j4_matched", "--run-name", f"{name}_seed{s}", "--init", INIT,
                                 "--eval-role", "test", "--panel", "--seed", str(s)] + extra,
                         "done": (lambda n=f"{name}_seed{s}": finished(f"{R}/runs/j4_matched/{n}/results.json", False))})
    return True


def next_job():
    with lock:
        try_choice()
        for j in JOBS:
            if j.get("state"):
                continue
            if j["done"]():
                j["state"] = "done"
                continue
            j["state"] = "running"
            return j
        return None


def worker(wid):
    while True:
        j = next_job()
        if j is None:
            with lock:
                if all(x.get("state") for x in JOBS) and os.path.exists(CHOICE):
                    return
            time.sleep(60)
            continue
        os.makedirs(os.path.dirname(j["log"]), exist_ok=True)
        t0 = time.time()
        print(f"[w{wid}] start {j['name']}", flush=True)
        with open(j["log"], "w") as log:
            rc = subprocess.call(j["cmd"], cwd=R, stdout=log, stderr=subprocess.STDOUT)
        with lock:
            j["state"] = "done" if rc == 0 else f"failed rc={rc}"
        print(f"[w{wid}] end {j['name']} rc={rc} {round((time.time() - t0) / 60, 1)} min", flush=True)


if __name__ == "__main__":
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in ts:
        t.start()
        time.sleep(30)
    for t in ts:
        t.join()
    print("QUEUE DONE", {j["name"]: j.get("state") for j in JOBS}, flush=True)

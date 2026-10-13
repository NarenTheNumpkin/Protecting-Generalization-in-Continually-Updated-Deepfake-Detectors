"""J3 job queue (registration/journal_J3.md). Two GPU workers; GPL test-split jobs wait for the development-data
learning-rate choice, which is made mechanically by the registered rule as soon as the three development runs exist."""

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
CHOICE = f"{R}/registration/J3_gpl_choice.json"
LONG = "FF,DFDCP#1,DFDCP#2,DFDCP#3,DFDCP#4,DFD#1,DFD#2,DFD#3,DFD#4,CDF"
MAIN = ["seq", "lwf", "featkd", "realkd", "featkd_lwf", "realkd_lwf"]


def done(tag, name):
    p = f"{R}/runs/{tag}/{name}/results.json"
    if not os.path.exists(p):
        return False
    r = json.load(open(p))
    return "panel" in r or r["args"].get("eval_role") == "dev" and len(r["stages"]) == len(r["args"]["stream"].split(","))


def job(tag, name, method, extra, needs_choice=False):
    return {"tag": tag, "name": name, "method": method, "extra": extra, "needs_choice": needs_choice}


def gpl_lr():
    return json.load(open(CHOICE))["lr"]


JOBS = [job("j3_gpl_dev", f"gpl_lr{lr}", "gpl", ["--lr", lr]) for lr in ["1e-4", "1e-3"]]
JOBS += [job("j3_gpl_dev", "seq_lr1e-4", "seq", [])]
TEST = ["--eval-role", "test", "--panel"]
JOBS += [job("j3_s1", "gpl_seed0", "gpl", TEST, True)]
JOBS += [job("j3_S2", "gpl_seed0", "gpl", TEST + ["--stream", "FF,DFDCP,CDF,DFD"], True),
         job("j3_S3", "gpl_seed0", "gpl", TEST + ["--stream", "FF,DFD,DFDCP,CDF"], True),
         job("j3_pe", "gpl_seed0", "gpl", TEST + ["--init", "yermandy/GenD_PE_L"], True)]
for s in [1, 2, 3, 4]:
    JOBS += [job("j3_seeds", f"{m}_seed{s}", m, TEST + ["--seed", str(s)]) for m in MAIN]
    JOBS += [job("j3_seeds", f"gpl_seed{s}", "gpl", TEST + ["--seed", str(s)], True)]
JOBS += [job("j3_long", f"{m}_seed0", m, TEST + ["--stream", LONG, "--eval-light"]) for m in ["seq", "lwf", "featkd_lwf", "realkd", "realkd_lwf"]]
JOBS += [job("j3_long", "gpl_seed0", "gpl", TEST + ["--stream", LONG, "--eval-light"], True)]


def gain_and_fsh(path):
    r = json.load(open(path))
    st, stream = r["stages"], r["args"]["stream"].split(",")
    gain = sum(st[j]["dev"][stream[j]]["auroc"] - st[j - 1]["dev"][stream[j]]["auroc"] for j in range(1, len(stream))) / (len(stream) - 1)
    return gain * 100, st[-1]["dev"]["FSh_probe"]["auroc"] * 100


def try_choice():
    """Registered rule: among candidates whose mean gain >= 0.5 x seq gain, the higher FSh-probe AUROC; else the higher gain."""
    if os.path.exists(CHOICE):
        return True
    paths = {k: f"{R}/runs/j3_gpl_dev/{k}/results.json" for k in ["gpl_lr1e-4", "gpl_lr1e-3", "seq_lr1e-4"]}
    if not all(done("j3_gpl_dev", k) for k in paths):
        return False
    m = {k: gain_and_fsh(p) for k, p in paths.items()}
    cand = {k: v for k, v in m.items() if k.startswith("gpl")}
    ok = {k: v for k, v in cand.items() if v[0] >= 0.5 * m["seq_lr1e-4"][0]}
    pick = max(ok, key=lambda k: ok[k][1]) if ok else max(cand, key=lambda k: cand[k][0])
    json.dump({"lr": pick.split("lr")[1], "rule": "registration/journal_J3.md block G", "dev": {k: {"gain": v[0], "FSh": v[1]} for k, v in m.items()},
               "qualified": sorted(ok)}, open(CHOICE, "w"), indent=1)
    print("GPL choice", open(CHOICE).read(), flush=True)
    return True


lock = threading.Lock()


def next_job():
    with lock:
        have_choice = try_choice()
        for j in JOBS:
            if j.get("state"):
                continue
            if done(j["tag"], j["name"]):
                j["state"] = "done"
                continue
            if j["needs_choice"] and not have_choice:
                continue
            j["state"] = "running"
            return j
        return None


def worker(wid):
    while True:
        j = next_job()
        if j is None:
            with lock:
                if all(x.get("state") for x in JOBS):
                    return
            time.sleep(60)
            continue
        extra = list(j["extra"])
        if j["method"] == "gpl" and j["tag"] != "j3_gpl_dev":
            extra += ["--lr", gpl_lr()]
        cmd = [PY, "stream.py", "--method", j["method"], "--tag", j["tag"], "--run-name", j["name"]] + extra
        if "--init" not in extra:
            cmd += ["--init", INIT]
        os.makedirs(f"{R}/runs/{j['tag']}", exist_ok=True)
        t0 = time.time()
        print(f"[w{wid}] start {j['tag']}/{j['name']}", flush=True)
        with open(f"{R}/runs/{j['tag']}/{j['name']}.log", "w") as log:
            rc = subprocess.call(cmd, cwd=R, stdout=log, stderr=subprocess.STDOUT)
        with lock:
            j["state"] = "done" if rc == 0 else f"failed rc={rc}"
        print(f"[w{wid}] end {j['tag']}/{j['name']} rc={rc} {round((time.time() - t0) / 60, 1)} min", flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
        time.sleep(20)
    for t in ts:
        t.join()
    try_choice()
    print("QUEUE DONE", {f"{j['tag']}/{j['name']}": j.get("state") for j in JOBS}, flush=True)

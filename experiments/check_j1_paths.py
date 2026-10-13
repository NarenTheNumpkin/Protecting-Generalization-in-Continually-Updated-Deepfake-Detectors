"""Focused integration check of the J1 code paths (no evaluation): every new method runs through stream.main()
with 1 epoch and 2+2 videos per task; scoring is stubbed. Checks: losses finite and the expected terms present,
only the intended parameters change, replay buffer growth, joint path, light-eval stage keys, PE backbone."""

import json
import shutil
import sys

import torch

import stream

stub = {"auroc": 0.5, "acc@0.5": 0.5, "real_fpr@0.5": 0.5, "fake_tpr@0.5": 0.5, "n_real": 1, "n_fake": 1}
stream.score = lambda *a, **k: dict(stub)
stream.score_panel = lambda m: ({"U": {"auroc": 0.5, "DFF_mean": 0.5}}, {})
OUTTAG = "_j1_check"
expect = {"featkd_lwf": {"kd", "kd_lwf"}, "fakekd": {"kd"}, "randkd": {"kd"}, "head": set(), "replay": set(),
          "joint": set(), "realkd_lwf": {"kd", "kd_lwf"}}
ok = True
for m, init in [(m, "yermandy/GenD_CLIP_L_14") for m in expect] + [("realkd_lwf", "yermandy/GenD_PE_L")]:
    name = f"{m}_{'pe' if 'PE' in init else 'clip'}"
    sys.argv = ["stream.py", "--method", m, "--epochs", "1", "--shots", "2", "--mem-videos", "1", "--eval-light",
                "--tag", OUTTAG, "--run-name", name, "--init", init, "--lam-lwf", "0.5"]
    stream.main()
    res = json.load(open(f"{stream.OUT}/{OUTTAG}/{name}/results.json"))
    st = res["stages"]
    terms = set().union(*[set(k for k in st[i]["train_log"][0] if k not in ("epoch", "sec", "gend")) for i in range(1, len(st))])
    acct = [s["train_log"][-1] for s in st[1:]]
    stage_keys = [sorted(s["dev"]) for s in st]
    finite = all(all(v == v and abs(v) < 1e6 for k, v in s["train_log"][0].items() if k not in ("epoch", "sec")) for s in st[1:])
    good = finite and terms == expect[m]
    if m == "head":
        good &= all(a["n_trainable"] == 2050 for a in acct)
    if m == "replay":
        good &= [s["n_train_videos"] for s in st[1:]] == [4 + 2, 4 + 4, 4 + 6]
    if m == "joint":
        good &= len(st) == 2 and st[1]["n_train_videos"] == 16
    ok &= good
    print(f"{name:16s} {'OK ' if good else 'FAIL'} terms={sorted(terms)} trainable={[a['n_trainable'] for a in acct]} "
          f"videos={[s['n_train_videos'] for s in st[1:]]} peakGiB={[a['peak_mem_gib'] for a in acct]} stage_keys={stage_keys}", flush=True)
shutil.rmtree(f"{stream.OUT}/{OUTTAG}")
print("ALL OK" if ok else "SOME FAILED")

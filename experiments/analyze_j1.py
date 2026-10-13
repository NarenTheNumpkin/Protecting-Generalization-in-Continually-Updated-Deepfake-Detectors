"""J1 analysis (registration/journal_J1.md). Works on whatever runs exist; missing runs are reported as such.

Outputs runs/j1_summary.json and prints: main table (S1), mechanism tests M1-M2, replication R1-R4 for S2/S3/PE,
sensitivity frontier points, data-regime table and cost table.
"""

import json
import os

import numpy as np

from analyze_c1 import GENS, U_from, boot_indices
from sklearn.metrics import roc_auc_score  # noqa: F401  (used through analyze_c1)

R = "runs"
T_S1 = ["FF", "DFDCP", "DFD", "CDF"]


def load(path):
    if not (os.path.exists(f"{path}/results.json") and os.path.exists(f"{path}/panel_preds.json")):
        return None
    res = json.load(open(f"{path}/results.json"))
    if "panel" not in res:
        return None
    args = json.load(open(f"{path}/args.json"))
    assert args["splits"] == "splits_v1.1" and args["eval_role"] == "test", path
    return {"res": res, "preds": json.load(open(f"{path}/panel_preds.json")), "args": args}


def metrics(run):
    """Stage-wise test metrics; works for full and light evaluation and for the joint oracle."""
    res, args = run["res"], run["args"]
    stream = args["stream"].split(",")
    st = res["stages"]
    out = {"U": res["panel"]["U"]["auroc"] * 100, "WDF": res["panel"]["WDF_aligned"]["auroc"] * 100,
           "DFF": res["panel"]["U"]["DFF_mean"] * 100, **{f"DFF_{g}": res["panel"][f"DFF_{g}"]["auroc"] * 100 for g in GENS},
           "WDF_prov": res["panel"]["WDF_provided"]["auroc"] * 100}
    fin = st[-1]["dev"]
    out["final"] = {t: fin[t]["auroc"] * 100 for t in T_S1}
    out["AA"] = float(np.mean(list(out["final"].values())))
    out["FF_acc"] = fin["FF"]["acc@0.5"] * 100
    out["AA_acc"] = float(np.mean([fin[t]["acc@0.5"] * 100 for t in T_S1]))
    out["real_fpr"] = {t: fin[t]["real_fpr@0.5"] * 100 for t in T_S1}
    out["FSh"] = fin["FSh_probe"]["auroc"] * 100
    if len(st) == len(stream):  # sequential runs: gain needs A[j-1][j] and A[j][j]
        A = lambda k, t: st[k]["dev"][t]["auroc"] * 100 if t in st[k]["dev"] else np.nan  # noqa: E731
        out["gain"] = float(np.mean([A(j, stream[j]) - A(j - 1, stream[j]) for j in range(1, len(stream))]))
        full = all(all(t in st[k]["dev"] for t in stream) for k in range(len(stream)))
        if full:
            K = len(stream) - 1
            out["AF"] = float(np.mean([max(A(k, stream[j]) for k in range(j, K)) - A(K, stream[j]) for j in range(K)]))
        cost = [s["train_log"][-1] for s in st[1:] if s.get("train_log") and "task_train_sec" in s["train_log"][-1]]
        if cost:
            out["train_min"] = sum(c["task_train_sec"] for c in cost) / 60
            out["peak_gib"] = max(c["peak_mem_gib"] for c in cost)
    return out


def paired_ci(a, b, reps=2000):
    PA, PB = a["preds"], b["preds"]
    for k in PA:
        assert PA[k]["video"] == PB[k]["video"], k
    rng = np.random.default_rng(20261005)
    d = [U_from(PA, ix) - U_from(PB, ix) for ix in (boot_indices(PA, rng) for _ in range(reps))]
    return float(U_from(PA) - U_from(PB)), [float(x) for x in np.percentile(d, [2.5, 97.5])]


def fmt(x, nd=2):
    return "  n/a " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:6.{nd}f}"


def main():
    summary = {}
    # ---------------- main table S1
    s1 = {m: load(f"{R}/c1/{m}_seed0") for m in ["none", "seq", "lwf", "featkd", "realkd", "realkd_lwf"]}
    s1 |= {m: load(f"{R}/j1_s1/{m}_seed0") for m in ["ewc", "lnmerge", "head", "featkd_lwf", "replay", "joint", "fakekd", "randkd"]}
    print("== Main table, S1 (clean init, seed 0); test AUROC, panel AUROC")
    print(f"{'method':11s} {'U':>6s} {'WDF':>6s} {'DFF':>6s} {'T2I':>6s} | {'FF':>6s} {'DFDCP':>6s} {'DFD':>6s} {'CDF':>6s} {'AA':>6s} {'AF':>6s} {'gain':>6s} {'FFacc':>6s}")
    summary["S1"] = {}
    for m, r in s1.items():
        if r is None:
            print(f"{m:11s} (not finished)")
            continue
        x = metrics(r)
        summary["S1"][m] = x
        print(f"{m:11s} {fmt(x['U'])} {fmt(x['WDF'])} {fmt(x['DFF'])} {fmt(x['DFF_text2img'])} | " + " ".join(fmt(x["final"][t]) for t in T_S1)
              + f" {fmt(x['AA'])} {fmt(x.get('AF'))} {fmt(x.get('gain'))} {fmt(x['FF_acc'], 1)}")
    # ---------------- mechanism tests
    print("\n== Mechanism (S1): difference in U and paired bootstrap 95% CI")
    summary["mechanism"] = {}
    for name, a, b in [("M1 realkd>randkd", "realkd", "randkd"), ("M2 realkd>fakekd", "realkd", "fakekd"),
                       ("A realkd_lwf>featkd_lwf", "realkd_lwf", "featkd_lwf")]:
        if s1.get(a) and s1.get(b):
            d, ci = paired_ci(s1[a], s1[b])
            summary["mechanism"][name] = {"diff": d, "ci": ci, "confirmed": d > 0 and ci[0] > 0}
            print(f"  {name:26s} {d:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] {'CONFIRMED' if d > 0 and ci[0] > 0 else 'not confirmed'}")
        else:
            print(f"  {name:26s} (pending)")
    # ---------------- replication
    print("\n== Replication R1-R4")
    summary["replication"] = {}
    for setting, tag in [("S2", "j1_S2"), ("S3", "j1_S3"), ("PE", "j1_pe")]:
        runs = {m: load(f"{R}/{tag}/{m}_seed0") for m in ["none", "seq", "lwf", "featkd", "realkd", "featkd_lwf", "realkd_lwf"]}
        summary["replication"][setting] = {"metrics": {m: metrics(r) for m, r in runs.items() if r}, "tests": {}}
        line = []
        for m, r in runs.items():
            if r:
                x = summary["replication"][setting]["metrics"][m]
                line.append(f"{m} U={x['U']:.2f} AA={x['AA']:.2f}")
        print(f"  {setting}: " + " | ".join(line))
        for name, a, b in [("R1", "realkd_lwf", "lwf"), ("R2", "realkd_lwf", "seq"), ("R3", "realkd", "featkd"), ("R4", "realkd_lwf", "featkd_lwf")]:
            if runs.get(a) and runs.get(b):
                d, ci = paired_ci(runs[a], runs[b])
                ok = d > 0 and ci[0] > 0
                summary["replication"][setting]["tests"][name] = {"a": a, "b": b, "diff": d, "ci": ci, "confirmed": ok}
                print(f"    {name} {a} > {b}: {d:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] {'CONFIRMED' if ok else 'not confirmed'}")
    # ---------------- sensitivity
    print("\n== Sensitivity (S1, light eval): U, AA, gain")
    summary["lambda"] = {}
    pts = {"realkd": [("1", s1.get("realkd"))], "featkd": [("1", s1.get("featkd"))], "realkd_lwf": [("g1_l1", s1.get("realkd_lwf"))]}
    for l in ["0.3", "3", "10"]:
        pts["realkd"].append((l, load(f"{R}/j1_lam/realkd_lam{l}")))
        pts["featkd"].append((l, load(f"{R}/j1_lam/featkd_lam{l}")))
    for gl in ["g0.3_l1", "g3_l1", "g1_l0.3", "g1_l3"]:
        pts["realkd_lwf"].append((gl, load(f"{R}/j1_lam/realkd_lwf_{gl}")))
    for fam, lst in pts.items():
        summary["lambda"][fam] = {}
        for lab, r in lst:
            if r:
                x = metrics(r)
                summary["lambda"][fam][lab] = {k: x[k] for k in ["U", "AA", "gain", "FF_acc", "WDF", "DFF"] if k in x}
                print(f"  {fam:11s} λ={lab:7s} U={x['U']:.2f} AA={x['AA']:.2f} gain={x.get('gain', float('nan')):+.2f} FFacc={x['FF_acc']:.1f}")
            else:
                print(f"  {fam:11s} λ={lab:7s} (pending)")
    # ---------------- data regime
    print("\n== Data regime (S1)")
    summary["shots"] = {}
    for k in [5, 10, 25]:
        for m in ["seq", "lwf", "realkd_lwf"]:
            r = s1.get(m) if k == 25 else load(f"{R}/j1_shots/{m}_shots{k}")
            if r:
                x = metrics(r)
                summary["shots"][f"{m}_{k}"] = {kk: x[kk] for kk in ["U", "AA", "gain", "FF_acc"] if kk in x}
                print(f"  shots={k:2d} {m:11s} U={x['U']:.2f} AA={x['AA']:.2f} gain={x.get('gain', float('nan')):+.2f}")
    # ---------------- cost (runs that logged it)
    print("\n== Cost per full stream (training only)")
    summary["cost"] = {}
    for tag in ["j1_S2", "j1_s1", "j1_pe"]:
        for m in ["seq", "lwf", "featkd", "realkd", "featkd_lwf", "realkd_lwf", "ewc", "lnmerge", "head", "replay", "fakekd", "randkd"]:
            r = load(f"{R}/{tag}/{m}_seed0")
            if r and m not in summary["cost"]:
                x = metrics(r)
                if "train_min" in x:
                    summary["cost"][m] = {"train_min": x["train_min"], "peak_gib": x["peak_gib"], "from": tag}
    for m, c in summary["cost"].items():
        print(f"  {m:11s} {c['train_min']:6.1f} min  peak {c['peak_gib']:.2f} GiB  ({c['from']})")
    json.dump(summary, open(f"{R}/j1_summary.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

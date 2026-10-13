"""J3 analysis (registration/journal_J3.md): blocks G (GPL comparison), S (five seeds), L (long stream).
Works on whatever runs exist. Writes runs/j3_summary.json."""

import json
import os

import numpy as np
from scipy.stats import ttest_rel

from analyze_c1 import U_from, boot_indices
from analyze_j1 import load, metrics

R = "runs"
T4 = ["FF", "DFDCP", "DFD", "CDF"]
MAIN = ["seq", "lwf", "featkd", "realkd", "featkd_lwf", "realkd_lwf", "gpl"]
SEED0 = {m: f"{R}/c1/{m}_seed0" for m in ["seq", "lwf", "featkd", "realkd", "realkd_lwf"]}
SEED0 |= {"featkd_lwf": f"{R}/j1_s1/featkd_lwf_seed0", "gpl": f"{R}/j3_s1/gpl_seed0", "none": f"{R}/c1/none_seed0"}


def path(m, s):
    return SEED0[m] if s == 0 else f"{R}/j3_seeds/{m}_seed{s}"


def paired_ci(a, b, reps=2000, seed=20261009):
    PA, PB = a["preds"], b["preds"]
    for k in PA:
        assert PA[k]["video"] == PB[k]["video"], k
    rng = np.random.default_rng(seed)
    d = [U_from(PA, ix) - U_from(PB, ix) for ix in (boot_indices(PA, rng) for _ in range(reps))]
    return float(U_from(PA) - U_from(PB)), [float(x) for x in np.percentile(d, [2.5, 97.5])]


def verdict(d, ci):
    return bool(d > 0 and ci[0] > 0)


def holm(pv, alpha=0.05):
    order = sorted(pv, key=pv.get)
    out, stop = {}, False
    for i, k in enumerate(order):
        adj = min(1.0, max(pv[j] * (len(order) - n) for n, j in enumerate(order[: i + 1])))
        rej = (not stop) and pv[k] <= alpha / (len(order) - i)
        stop = stop or not rej
        out[k] = {"p": pv[k], "p_holm": adj, "reject": rej}
    return out


def long_metrics(run):
    res = run["res"]
    st, stream = res["stages"], run["args"]["stream"].split(",")
    fin = st[-1]["dev"]
    out = {"U": res["panel"]["U"]["auroc"] * 100, "WDF": res["panel"]["WDF_aligned"]["auroc"] * 100, "DFF": res["panel"]["U"]["DFF_mean"] * 100,
           "DFF_text2img": res["panel"]["DFF_text2img"]["auroc"] * 100, "final": {t: fin[t]["auroc"] * 100 for t in T4}}
    out["AA"] = float(np.mean(list(out["final"].values())))
    out["FF_acc"] = fin["FF"]["acc@0.5"] * 100
    gains = []
    for j in range(1, len(stream)):
        t = stream[j].split("#")[0]
        if t in st[j]["dev"] and t in st[j - 1]["dev"]:
            gains.append((st[j]["dev"][t]["auroc"] - st[j - 1]["dev"][t]["auroc"]) * 100)
    out["gain"] = float(np.mean(gains))
    cost = [s["train_log"][-1]["task_train_sec"] for s in st[1:] if s.get("train_log")]
    out["train_min"] = sum(cost) / 60
    return out


def main():
    S = {}
    # ---------------- G: GPL vs GAD+LwF in four settings (seed 0)
    print("== G: GPL re-implemented on GenD (seed 0)")
    choice = json.load(open("registration/J3_gpl_choice.json")) if os.path.exists("registration/J3_gpl_choice.json") else None
    print("  lr choice:", choice)
    S["G"] = {"choice": choice, "tests": {}, "metrics": {}}
    for name, setting, ours, gpl in [("G1", "S1", f"{R}/c1/realkd_lwf_seed0", f"{R}/j3_s1/gpl_seed0"),
                                     ("G2", "S2", f"{R}/j1_S2/realkd_lwf_seed0", f"{R}/j3_S2/gpl_seed0"),
                                     ("G3", "S3", f"{R}/j1_S3/realkd_lwf_seed0", f"{R}/j3_S3/gpl_seed0"),
                                     ("G4", "PE", f"{R}/j1_pe/realkd_lwf_seed0", f"{R}/j3_pe/gpl_seed0")]:
        a, b = load(ours), load(gpl)
        if not (a and b):
            print(f"  {name} {setting}: pending")
            continue
        ma, mb = metrics(a), metrics(b)
        d, ci = paired_ci(a, b)
        S["G"]["metrics"][setting] = {"realkd_lwf": ma, "gpl": mb}
        S["G"]["tests"][name] = {"setting": setting, "diff": d, "ci": ci, "confirmed": verdict(d, ci)}
        print(f"  {name} {setting}: U GAD+LwF {ma['U']:.2f} vs GPL {mb['U']:.2f}  diff {d:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] "
              f"{'CONFIRMED' if verdict(d, ci) else 'NOT CONFIRMED'} | AA {ma['AA']:.2f} vs {mb['AA']:.2f} | gain {ma.get('gain', float('nan')):+.2f} vs "
              f"{mb.get('gain', float('nan')):+.2f} | T2I {ma['DFF_text2img']:.2f} vs {mb['DFF_text2img']:.2f} | ACC_FF {ma['FF_acc']:.1f} vs {mb['FF_acc']:.1f}")
    # ---------------- S: five seeds
    print("\n== S: five seeds, S1 (mean ± SD)")
    runs = {m: {s: load(path(m, s)) for s in range(5)} for m in MAIN}
    S["S"] = {"per_seed": {}, "summary": {}, "tests": {}}
    for m in MAIN:
        ms = {s: metrics(r) for s, r in runs[m].items() if r}
        S["S"]["per_seed"][m] = {s: {k: v for k, v in x.items() if k in ("U", "AA", "AF", "gain", "FF_acc", "DFF_text2img", "WDF", "DFF")} for s, x in ms.items()}
        if not ms:
            continue
        summ = {k: (float(np.mean([x[k] for x in ms.values()])), float(np.std([x[k] for x in ms.values()], ddof=1)) if len(ms) > 1 else float("nan"))
                for k in ["U", "AA", "AF", "gain", "FF_acc", "DFF_text2img"]}
        S["S"]["summary"][m] = {"n": len(ms), **summ}
        print(f"  {m:11s} n={len(ms)} " + " ".join(f"{k} {v[0]:.2f}±{v[1]:.2f}" for k, v in summ.items()))
    pv = {}
    for name, a, b in [("S-H1", "realkd_lwf", "lwf"), ("S-H2", "realkd_lwf", "seq"), ("S-H3", "realkd", "featkd"),
                       ("S-H4", "realkd_lwf", "featkd_lwf"), ("S-H5", "realkd_lwf", "gpl")]:
        seeds = [s for s in range(5) if runs[a][s] and runs[b][s]]
        if len(seeds) < 2:
            print(f"  {name}: pending")
            continue
        d = [metrics(runs[a][s])["U"] - metrics(runs[b][s])["U"] for s in seeds]
        p = float(ttest_rel([metrics(runs[a][s])["U"] for s in seeds], [metrics(runs[b][s])["U"] for s in seeds], alternative="greater").pvalue)
        boots = {s: paired_ci(runs[a][s], runs[b][s]) for s in seeds}
        S["S"]["tests"][name] = {"a": a, "b": b, "seeds": seeds, "diffs": d, "mean": float(np.mean(d)), "sd": float(np.std(d, ddof=1)),
                                 "p": p, "n_pos": int(sum(x > 0 for x in d)), "boot_ci": {s: v[1] for s, v in boots.items()},
                                 "boot_excl0": int(sum(v[1][0] > 0 for v in boots.values()))}
        pv[name] = p
    for name, h in holm(pv).items():
        t = S["S"]["tests"][name]
        t |= {"p_holm": h["p_holm"], "confirmed": bool(t["mean"] > 0 and h["reject"] and len(t["seeds"]) == 5)}
        print(f"  {name} {t['a']} > {t['b']}: mean ΔU {t['mean']:+.2f} ± {t['sd']:.2f} (n={len(t['seeds'])}), positive in {t['n_pos']}, "
              f"bootstrap CI excl. 0 in {t['boot_excl0']}; p={t['p']:.4f}, p_holm={t['p_holm']:.4f} -> {'CONFIRMED' if t['confirmed'] else 'not confirmed'}")
    # ---------------- L: long stream
    print("\n== L: nine-update stream (seed 0)")
    L = {m: load(f"{R}/j3_long/{m}_seed0") for m in ["seq", "lwf", "featkd_lwf", "realkd", "realkd_lwf", "gpl"]}
    S["L"] = {"metrics": {}, "tests": {}}
    for m, r in L.items():
        if r:
            x = long_metrics(r)
            S["L"]["metrics"][m] = x
            print(f"  {m:11s} U {x['U']:.2f} T2I {x['DFF_text2img']:.2f} AA {x['AA']:.2f} final {({k: round(v, 2) for k, v in x['final'].items()})} gain {x['gain']:+.2f} "
                  f"FFacc {x['FF_acc']:.1f} train {x['train_min']:.0f} min")
    for name, a, b in [("L1", "realkd_lwf", "lwf"), ("L2", "realkd_lwf", "seq"), ("L3", "realkd_lwf", "gpl")]:
        if L.get(a) and L.get(b):
            d, ci = paired_ci(L[a], L[b])
            S["L"]["tests"][name] = {"a": a, "b": b, "diff": d, "ci": ci, "confirmed": verdict(d, ci)}
            print(f"  {name} {a} > {b}: {d:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] {'CONFIRMED' if verdict(d, ci) else 'NOT CONFIRMED'}")
    json.dump(S, open(f"{R}/j3_summary.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

"""C1 analysis as registered (registration/confirmation_C1.md + Amendment C1-A: seed 0 only).

Primary U = mean(WDF_aligned AUROC, mean over 3 DFF generators AUROC), after the last stage.
C-H1 M* > lwf, C-H2 M* > seq, C-H3 realkd > featkd: seed-0 difference > 0 AND paired bootstrap 95% CI excluding 0
(2,000 resamples; WildDeepfake resampled by source archive within class, DeepFakeFace by image within class).
"""

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

R = "runs/c1"
T = ["FF", "DFDCP", "DFD", "CDF"]
M_STAR = json.load(open("registration/C1_choice.json"))["M_star"]
METHODS = ["none", M_STAR, "lwf", "seq", "realkd", "featkd"]
GENS = ["inpainting", "insight", "text2img"]


def load(m):
    d = f"{R}/{m}_seed0"
    args = json.load(open(f"{d}/args.json"))
    assert args["splits"] == "splits_v1.1" and args["eval_role"] == "test" and args["panel"], d
    assert args["init"].endswith("best_mAP.ckpt"), d
    return json.load(open(f"{d}/results.json")), json.load(open(f"{d}/panel_preds.json"))


def stream_metrics(res):
    st = res["stages"]
    n = len(st)
    A = np.array([[st[i]["dev"][t]["auroc"] for t in T] for i in range(n)]) * 100  # key "dev" holds the eval role
    acc = np.array([[st[i]["dev"][t]["acc@0.5"] for t in T] for i in range(n)]) * 100
    out = {"AA": A[-1].mean(), "FF_acc": acc[-1, 0], "AA_acc": acc[-1].mean(), "final": dict(zip(T, A[-1].round(2))),
           "FSh": st[-1]["dev"]["FSh_probe"]["auroc"] * 100}
    if n == 4:
        out["AF"] = np.mean([A[j:3, j].max() - A[3, j] for j in range(3)])
        out["gain"] = np.mean([A[j, j] - A[j - 1, j] for j in range(1, 4)])
    return out


def U_from(preds, idx=None):
    """idx: dict of index arrays per panel set (bootstrap), else full."""
    def auc(name):
        y = np.array(preds[name]["label"]); p = np.array(preds[name]["p_fake"])
        if idx is not None:
            y, p = y[idx[name]], p[idx[name]]
        return roc_auc_score(y, p)
    wdf = auc("WDF_aligned")
    dff = np.mean([auc(f"DFF_{g}") for g in GENS])
    return 100 * (wdf + dff) / 2


def boot_indices(preds, rng):
    idx = {}
    # WildDeepfake: clusters = source archive id (unit "<archive>_<seq>"), resampled within class
    for name in ["WDF_aligned", "WDF_provided"]:
        v = preds[name]["video"]; y = np.array(preds[name]["label"])
        arc = np.array([os.path.basename(x).split("_")[0] + f"/{lab}" for x, lab in zip(v, y)])
        take = []
        for lab in [0, 1]:
            cl = np.unique(arc[y == lab])
            for c in rng.choice(cl, len(cl), replace=True):
                take.append(np.flatnonzero(arc == c))
        idx[name] = np.concatenate(take)
    # DeepFakeFace: same real images in all three generator sets -> resample reals once, fakes per generator
    real_names = [os.path.basename(x) for x, l in zip(preds["DFF_inpainting"]["video"], preds["DFF_inpainting"]["label"]) if l == 0]
    real_pick = rng.choice(real_names, len(real_names), replace=True)
    for g in GENS:
        v = [os.path.basename(x) for x in preds[f"DFF_{g}"]["video"]]; y = np.array(preds[f"DFF_{g}"]["label"])
        pos = {n: i for i, (n, l) in enumerate(zip(v, y)) if l == 0}
        fakes = np.flatnonzero(y == 1)
        idx[f"DFF_{g}"] = np.concatenate([[pos[n] for n in real_pick], rng.choice(fakes, len(fakes), replace=True)])
    return idx


def paired(a, b, PA, PB, reps=2000):
    for k in PA:  # identical unit ordering is required for pairing
        assert PA[k]["video"] == PB[k]["video"], k
    rng = np.random.default_rng(20261004)
    d = [U_from(PA, ix) - U_from(PB, ix) for ix in (boot_indices(PA, rng) for _ in range(reps))]
    return U_from(PA) - U_from(PB), np.percentile(d, [2.5, 97.5])


def main():
    res, preds = {}, {}
    for m in METHODS:
        res[m], preds[m] = load(m)
    print(f"M* = {M_STAR}   (single seed 0, Amendment C1-A)\n")
    print(f"{'method':11s} {'U':>6s} {'WDF':>6s} {'WDFprov':>7s} {'DFF':>6s} {'inpaint':>7s} {'insight':>7s} {'txt2img':>7s} | {'test AA':>7s} {'AF':>5s} {'gain':>6s} {'FFacc':>6s} {'AAacc':>6s} {'FSh':>6s}")
    sm = {}
    for m in METHODS:
        p = res[m]["panel"]; sm[m] = stream_metrics(res[m])
        s = sm[m]
        print(f"{m:11s} {p['U']['auroc']*100:6.2f} {p['WDF_aligned']['auroc']*100:6.2f} {p['WDF_provided']['auroc']*100:7.2f} {p['U']['DFF_mean']*100:6.2f} "
              + " ".join(f"{p[f'DFF_{g}']['auroc']*100:7.2f}" for g in GENS)
              + f" | {s['AA']:7.2f} {s.get('AF', float('nan')):5.2f} {s.get('gain', float('nan')):+6.2f} {s['FF_acc']:6.1f} {s['AA_acc']:6.1f} {s['FSh']:6.2f}")
    print("\nfinal test AUROC per task:")
    for m in METHODS:
        print(f"  {m:11s} {sm[m]['final']}")
    print("\nConfirmatory hypotheses (diff > 0 and 95% CI excluding 0):")
    verdicts = {}
    for name, a, b in [("C-H1", M_STAR, "lwf"), ("C-H2", M_STAR, "seq"), ("C-H3", "realkd", "featkd")]:
        d, ci = paired(a, b, preds[a], preds[b])
        ok = d > 0 and ci[0] > 0
        verdicts[name] = bool(ok)
        print(f"  {name}: U({a}) - U({b}) = {d:+.2f}  CI [{ci[0]:+.2f}, {ci[1]:+.2f}]  -> {'CONFIRMED' if ok else 'NOT CONFIRMED'}")
    others = [m for m in METHODS if m not in ("none", M_STAR)]
    best_aa = max(sm[m]["AA"] for m in others)
    print(f"\nGuardrails for {M_STAR}: test AA {sm[M_STAR]['AA']:.2f} >= {best_aa:.2f} - 0.5 -> {'pass' if sm[M_STAR]['AA'] >= best_aa - 0.5 else 'FAIL'};"
          f" gain {sm[M_STAR]['gain']:+.2f} >= 0.5 x seq {sm['seq']['gain']:+.2f} -> {'pass' if sm[M_STAR]['gain'] >= 0.5 * sm['seq']['gain'] else 'FAIL'}")
    print("Retention vs no-update (U):", {m: round(res[m]['panel']['U']['auroc'] * 100 - res['none']['panel']['U']['auroc'] * 100, 2) for m in METHODS if m != "none"})
    json.dump({"verdicts": verdicts, "M_star": M_STAR}, open(f"{R}/c1_verdicts.json", "w"), indent=1)


if __name__ == "__main__":
    main()

"""Gate 2 analysis exactly as registered in dev_protocol_v1.md Amendment A2 (dev splits, splits_v1.1)."""
import json
import numpy as np

T = ["FF", "DFDCP", "DFD", "CDF"]
RUNS = {"seq": "gate1_v1.1", **{m: "gate2" for m in ["radial", "sdc", "featkd", "lwf", "ewc", "lnmerge"]}}


def metrics(tag, m, s):
    run = f"runs/{tag}/{m}_seed{s}"
    assert json.load(open(f"{run}/args.json"))["splits"] == "splits_v1.1", run
    st = json.load(open(f"{run}/results.json"))["stages"]
    A = np.array([[st[i]["dev"][t]["auroc"] for t in T] for i in range(4)]) * 100
    acc = np.array([[st[i]["dev"][t]["acc@0.5"] for t in T] for i in range(4)]) * 100
    fsh = np.array([st[i]["dev"]["FSh_probe"]["auroc"] for i in range(4)]) * 100
    return {
        "FSh_final": fsh[3], "AA": A[3].mean(), "AF": np.mean([A[j:3, j].max() - A[3, j] for j in range(3)]),
        "gain": np.mean([A[j, j] - A[j - 1, j] for j in range(1, 4)]), "FF_acc": acc[3, 0], "AA_acc": acc[3].mean(),
        "base": (fsh[0], A[0].mean(), acc[0, 0], acc[0].mean()),
    }


R = {m: [metrics(tag, m, s) for s in [0, 1, 2]] for m, tag in RUNS.items()}
mean = lambda m, k: float(np.mean([r[k] for r in R[m]]))  # noqa: E731
b = R["seq"][0]["base"]
print(f"{'method':9s} {'FSh final':>10s} {'per-seed FSh':>22s} {'AA':>6s} {'AF':>5s} {'gain':>6s} {'FF ACC':>7s} {'AA ACC':>7s}")
print(f"{'no-update':9s} {b[0]:10.2f} {'':>22s} {b[1]:6.2f} {'':>5s} {'':>6s} {b[2]:7.1f} {b[3]:7.1f}")
for m in R:
    print(f"{m:9s} {mean(m,'FSh_final'):10.2f} {str([round(r['FSh_final'],2) for r in R[m]]):>22s} {mean(m,'AA'):6.2f} {mean(m,'AF'):5.2f} {mean(m,'gain'):+6.2f} {mean(m,'FF_acc'):7.1f} {mean(m,'AA_acc'):7.1f}")

controls = [m for m in R if m != "radial"]
best = max(controls, key=lambda m: mean(m, "FSh_final"))
diffs = [R["radial"][s]["FSh_final"] - R[best][s]["FSh_final"] for s in range(3)]
best_aa = max(mean(m, "AA") for m in controls)
p1 = mean("radial", "FSh_final") > mean(best, "FSh_final") and all(d > 0 for d in diffs)
g1 = mean("radial", "AA") >= best_aa - 0.5
g2 = mean("radial", "gain") >= 0.5 * mean("seq", "gain")
print(f"\nPrimary: RADIAL FSh {mean('radial','FSh_final'):.2f} vs best control {best} {mean(best,'FSh_final'):.2f}; paired diffs {np.round(diffs,2)} -> {'PASS' if p1 else 'FAIL'}")
print(f"Guardrail AA: {mean('radial','AA'):.2f} >= {best_aa:.2f} - 0.5 -> {'PASS' if g1 else 'FAIL'}")
print(f"Guardrail gain: {mean('radial','gain'):+.2f} >= 0.5 x seq {mean('seq','gain'):+.2f} -> {'PASS' if g2 else 'FAIL'}")
print("GATE 2:", "PASS" if (p1 and g1 and g2) else "FAIL")
for abl in ["featkd", "sdc"]:
    d = [R["radial"][s]["FSh_final"] - R[abl][s]["FSh_final"] for s in range(3)]
    print(f"Ablation radial - {abl}: FSh paired diffs {np.round(d,2)} (mean {np.mean(d):+.2f}); AA diff {mean('radial','AA')-mean(abl,'AA'):+.2f}")

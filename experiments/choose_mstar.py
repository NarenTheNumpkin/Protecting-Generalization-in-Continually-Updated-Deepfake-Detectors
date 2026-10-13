"""Apply the E2 decision rule (registration/exploratory_E2.md) and record M* for C1."""
import json, numpy as np
T = ["FF", "DFDCP", "DFD", "CDF"]
def m(tag, meth):
    fs, aa = [], []
    for s in range(3):
        st = json.load(open(f"runs/{tag}/{meth}_seed{s}/results.json"))["stages"]
        fs.append(st[3]["dev"]["FSh_probe"]["auroc"] * 100)
        aa.append(np.mean([st[3]["dev"][t]["auroc"] for t in T]) * 100)
    return float(np.mean(fs)), float(np.mean(aa)), fs
f2, a2, per2 = m("e2", "realkd_lwf")
f1, a1, per1 = m("e1", "radial_lwf")
ok = (f2 >= f1 - 1.0) and (a2 >= a1 - 0.5)
choice = {"M_star": "realkd_lwf" if ok else "radial_lwf", "realkd_lwf": {"FSh": f2, "AA": a2, "FSh_per_seed": per2},
          "radial_lwf": {"FSh": f1, "AA": a1, "FSh_per_seed": per1}, "rule": "FSh within 1.0 and AA within 0.5 of radial_lwf"}
json.dump(choice, open("registration/C1_choice.json", "w"), indent=1)
print(json.dumps(choice))

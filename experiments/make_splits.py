"""Register the continual-learning task stream and its identity-disjoint splits (written once, before any
continual training). Roles per task:
  fit   - pool from which incremental training videos are drawn (few-shot: 25 real + 25 fake per seed)
  dev   - development evaluation (all method/hyperparameter decisions)
  test  - sealed until final runs (official test lists where they exist)

Identity rules (a video's identities = every person id in its name/metadata):
  FF++  : official train/val/test pair splits (base task, learned by the GenD checkpoint).
  DFDCP : official train/test; train identities that also occur in test are dropped, the remaining
          train identities are split fit/dev; a clip joins a role only if all its identities are in it.
  DFD   : no official split exists (DeepfakeBench puts every DFD video in train, val and test), so the
          28 actors are split fit/dev/test; a fake A_B joins a role only if both A and B are in it.
  CDF   : official 518-video test list. Every test identity also appears in non-test videos, so
          fit = non-test videos with no test identity (+ YouTube-real); dev = non-test videos whose
          identities are all test identities (test videos themselves stay sealed).
  FSh   : FaceShifter (never trained on) - unseen-forgery development probe: FF++ val split.
"""

import paths
import json
import os
import random
import re
from collections import Counter

SEED = 20261003
RAW = paths.RAW_DATA
FFO = f"{RAW}/FaceForensics_official"
OUT = os.path.join(paths.REGISTRATION, "splits_v1.json")


def split_ids(ids, fractions, rng):
    ids = sorted(ids)
    rng.shuffle(ids)
    out, start = {}, 0
    for i, (role, frac) in enumerate(fractions):
        end = len(ids) if i == len(fractions) - 1 else start + round(frac * len(ids))
        out.update({x: role for x in ids[start:end]})
        start = end
    return out


def assign(videos, role_of_id):
    """videos: {video: (label, [ids])} -> {role: {'real': [...], 'fake': [...]}}; drop cross-role videos."""
    roles, dropped = {}, 0
    for v, (label, ids) in sorted(videos.items()):
        r = {role_of_id.get(i) for i in ids}
        if len(r) != 1 or None in r:
            dropped += 1
            continue
        roles.setdefault(r.pop(), {"real": [], "fake": []})[label].append(v)
    return roles, dropped


def ffpp():
    sp = {k: json.load(open(os.path.join(paths.FF_SPLITS, f"{k}.json"))) for k in ["train", "val", "test"]}
    role_name = {"train": "fit", "val": "dev", "test": "test"}
    out = {}
    for k, pairs in sp.items():
        real = sorted({v for p in pairs for v in p})
        fake = sorted({f"{a}_{b}" for a, b in pairs} | {f"{b}_{a}" for a, b in pairs})
        out[role_name[k]] = {"real": [f"real/{v}" for v in real], "fake": [f"{m}/{v}" for m in ["DF", "F2F", "FS", "NT"] for v in fake]}
    return out


def dfdcp(rng):
    meta = json.load(open(f"{RAW}/../external/DFDC-preview-kaggle-frames/dataset.json"))
    have = set()
    for m in ["original_videos", "method_A", "method_B"]:
        for f in os.listdir(f"{RAW}/../external/DFDC-preview-kaggle-frames/{m}/frames"):
            have.add((m, f))
    vids = {"train": {}, "test": {}}
    for key, e in meta.items():
        method = key.split("/")[0]
        folder = os.path.basename(key)[:-4]
        if (method, folder) not in have:
            continue
        src = {"original_videos": "real", "method_A": "method_A", "method_B": "method_B"}[method]
        ids = [e["target_id"]] + ([e["swapped_id"]] if e["swapped_id"] else [])
        vids[e["set"]][f"{src}/{folder}"] = ("real" if e["label"] == "real" else "fake", ids)
    test_ids = {i for _, ids in vids["test"].values() for i in ids}
    train_ids = {i for _, ids in vids["train"].values() for i in ids} - test_ids
    role_of = split_ids(train_ids, [("fit", 0.7), ("dev", 0.3)], rng)
    roles, dropped = assign(vids["train"], role_of)
    roles["test"] = {"real": sorted(v for v, (l, _) in vids["test"].items() if l == "real"),
                     "fake": sorted(v for v, (l, _) in vids["test"].items() if l == "fake")}
    return roles, {"dropped_cross_role_or_test_identity": dropped, "test_identities_excluded_from_train": len({i for _, ids in vids["train"].values() for i in ids} & test_ids)}


def dfd(rng):
    real = [f[:-4] for f in os.listdir(f"{FFO}/original_sequences/actors/c23/videos") if f.endswith(".mp4")]
    fake = [f[:-4] for f in os.listdir(f"{FFO}/manipulated_sequences/DeepFakeDetection/c23/videos") if f.endswith(".mp4")]
    vids = {f"real/{v}": ("real", [v.split("__")[0]]) for v in real}
    vids |= {f"fake/{v}": ("fake", v.split("__")[0].split("_")) for v in fake}
    actors = {a for _, ids in vids.values() for a in ids}
    role_of = split_ids(actors, [("fit", 8 / 28), ("dev", 5 / 28), ("test", 15 / 28)], rng)
    roles, dropped = assign(vids, role_of)
    return roles, {"actors": {r: sorted(a for a, x in role_of.items() if x == r) for r in ["fit", "dev", "test"]}, "dropped_cross_role": dropped}


def cdf():
    test = [l.split()[1][:-4] for l in open(f"{RAW}/Celeb-DF-v2/List_of_testing_videos.txt").read().strip().split("\n")]
    alls = [f"{s}/{f[:-4]}" for s in ["Celeb-real", "Celeb-synthesis", "YouTube-real"] for f in os.listdir(f"{RAW}/Celeb-DF-v2/{s}")]
    nontest = sorted(set(alls) - set(test))
    ids = lambda v: set(re.findall(r"id\d+", v.split("/")[1]))  # noqa: E731
    test_ids = set().union(*map(ids, test))
    label = lambda v: "fake" if v.startswith("Celeb-synthesis") else "real"  # noqa: E731
    fit = [v for v in nontest if not (ids(v) & test_ids)]
    dev = [v for v in nontest if ids(v) and ids(v) <= test_ids]
    pack = lambda vs: {"real": [v for v in vs if label(v) == "real"], "fake": [v for v in vs if label(v) == "fake"]}  # noqa: E731
    return {"fit": pack(fit), "dev": pack(dev), "test": pack(sorted(test))}, {"test_identities": len(test_ids)}


def fsh():
    pairs = json.load(open(os.path.join(paths.FF_SPLITS, "val.json")))
    fake = sorted({f"{a}_{b}" for a, b in pairs} | {f"{b}_{a}" for a, b in pairs})
    return {"dev": {"real": [], "fake": [f"FSh/{v}" for v in fake]}}, {"note": "reals for the FSh probe = FF++ dev reals"}


def main():
    rng = random.Random(SEED)
    tasks, notes = {}, {}
    tasks["FF"] = ffpp()
    tasks["DFDCP"], notes["DFDCP"] = dfdcp(rng)
    tasks["DFD"], notes["DFD"] = dfd(rng)
    tasks["CDF"], notes["CDF"] = cdf()
    tasks["FSh_probe"], notes["FSh_probe"] = fsh()
    reg = {
        "version": "splits_v1",
        "seed": SEED,
        "stream_S1": ["FF", "DFDCP", "DFD", "CDF"],
        "few_shot": {"real": 25, "fake": 25, "seeds": [0, 1, 2]},
        "tasks": tasks,
        "notes": notes,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    assert not os.path.exists(OUT), f"{OUT} exists; splits are registered once"
    json.dump(reg, open(OUT, "w"), indent=1)
    for t, roles in tasks.items():
        print(t, {r: {k: len(v) for k, v in d.items()} for r, d in roles.items()}, notes.get(t, ""))


if __name__ == "__main__":
    main()

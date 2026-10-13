"""Continual adaptation of GenD over a task stream (default S1: FF -> DFDCP -> DFD -> CDF).

Methods (all tune only LayerNorm affine params + the LinearNorm head, starting from a GenD checkpoint
that has already learned FF):
  seq     naive sequential fine-tuning with the GenD loss (lower bound; Gate 1)
  lwf     + logit distillation from the previous model on current data (Li & Hoiem)
  ewc     + diagonal-Fisher quadratic penalty, Fisher estimated at the end of each task (Kirkpatrick et al.)
  featkd  + symmetric feature distillation 1-cos(z, z_prev) on ALL current samples (control for radial)
  lnmerge seq per task, then running average of LN/head params across tasks (model-merging control)
  sdc     + vMF prototype replay for the head with SDC drift compensation from ALL current samples
  radial  + asymmetric REAL-ONLY feature distillation + vMF prototype replay whose drift is estimated
          from current genuine frames for real prototypes (all frames for fake prototypes)
Memory for prototype methods: K spherical k-means centroids + vMF concentration per (task, class); no images.

Evaluation after every stage: dev split of every task + the FaceShifter probe (never trained),
video-level AUROC and ACC@0.5 (8 evenly spaced frames per video on dev).
"""

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import FrameSet, embed, few_shot, frames, gend_loss, load_model, load_registration, score, trainable_state, videos

OUT = "/root/deepfake-research/radial/runs"


# ----------------------------------------------------------------------------- prototype memory

def spherical_kmeans(z, k, iters=30, seed=0):
    # farthest-point initialisation: start from a seeded random point, then repeatedly add the point
    # least similar to all chosen centroids (avoids two seeds inside one cluster)
    g = torch.Generator(device=z.device).manual_seed(seed)
    idx = [int(torch.randint(len(z), (1,), generator=g, device=z.device))]
    for _ in range(k - 1):
        idx.append(int((z @ z[idx].T).max(1).values.argmin()))
    c = z[idx].clone()
    for _ in range(iters):
        a = (z @ c.T).argmax(1)
        for j in range(len(c)):
            m = z[a == j]
            if len(m):
                c[j] = F.normalize(m.mean(0), dim=0)
    a = (z @ c.T).argmax(1)
    return c, a


def vmf_kappa(rbar, d):
    rbar = min(max(rbar, 1e-4), 1 - 1e-4)
    return rbar * (d - rbar ** 2) / (1 - rbar ** 2)  # Banerjee et al. 2005 approximation


def sample_vmf(mu, kappa, n):
    """Wood (1994) sampler for vMF on S^{d-1}."""
    d = mu.shape[0]
    b = (-2 * kappa + math.sqrt(4 * kappa ** 2 + (d - 1) ** 2)) / (d - 1)
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * math.log(1 - x0 ** 2)
    ws = []
    beta = torch.distributions.Beta((d - 1) / 2, (d - 1) / 2)
    while len(ws) < n:
        zz = beta.sample((n,))
        w = (1 - (1 + b) * zz) / (1 - (1 - b) * zz)
        u = torch.rand(n)
        ok = kappa * w + (d - 1) * torch.log(1 - x0 * w) - c >= torch.log(u)
        ws += w[ok].tolist()
    w = torch.tensor(ws[:n], device=mu.device).unsqueeze(1)
    v = torch.randn(n, d, device=mu.device)
    v = F.normalize(v - (v @ mu).unsqueeze(1) * mu, dim=1)
    return w * mu + torch.sqrt(1 - w ** 2) * v


@torch.no_grad()
def extract(model, video_list, per_video=8, batch=128):
    model.eval()
    items = frames(video_list, per_video)
    dl = DataLoader(FrameSet(items, model.feature_extractor.preprocess, False), batch_size=batch, num_workers=8)
    zs, ys = [], []
    for x, y, _ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, z = embed(model, x.cuda(non_blocking=True))
        zs.append(z.float())
        ys.append(y.cuda())
    return torch.cat(zs), torch.cat(ys), items


def build_memory(z, y, k):
    mem = []
    for lab in [0, 1]:
        zl = z[y == lab]
        c, a = spherical_kmeans(zl, min(k, len(zl)))
        for j in range(len(c)):
            m = zl[a == j]
            if len(m) < 2:
                continue
            rbar = float(m.mean(0).norm())
            mem.append({"mu": c[j].clone(), "kappa": vmf_kappa(rbar, z.shape[1]), "label": lab, "n": len(m)})
    return mem


def compensate(mem, z_old, z_new, y, real_anchor: bool):
    """SDC-style drift: Delta_k = sum_i w_ik (z_new_i - z_old_i) / sum_i w_ik, w_ik = exp(-|z_old_i - mu_k|^2 / 2s^2).
    real_anchor=True: real prototypes use only genuine frames; fake prototypes use all frames."""
    delta = z_new - z_old
    for m in mem:
        sel = (y == 0) if (real_anchor and m["label"] == 0) else torch.ones_like(y, dtype=torch.bool)
        zo, dl = z_old[sel], delta[sel]
        d2 = ((zo - m["mu"]) ** 2).sum(1)
        s2 = d2.median().clamp(min=1e-6)
        w = torch.exp(-d2 / (2 * s2))
        m["mu"] = F.normalize(m["mu"] + (w.unsqueeze(1) * dl).sum(0) / w.sum().clamp(min=1e-8), dim=0)


def replay_batch(mem, n):
    weights = torch.tensor([m["n"] for m in mem], dtype=torch.float)
    # balance classes, then proportional to cluster size within class
    for lab in [0, 1]:
        idx = [i for i, m in enumerate(mem) if m["label"] == lab]
        weights[idx] = weights[idx] / weights[idx].sum()
    pick = torch.multinomial(weights, n, replacement=True)
    z = torch.cat([sample_vmf(mem[i]["mu"], mem[i]["kappa"], 1) for i in pick.tolist()])
    y = torch.tensor([mem[i]["label"] for i in pick.tolist()], device=z.device)
    return z, y


# ----------------------------------------------------------------------------- fisher

def fisher_diag(model, video_list, n_batches=40, batch=16):
    model.train()
    items = frames(video_list, 8)
    dl = DataLoader(FrameSet(items, model.feature_extractor.preprocess, False), batch_size=batch, shuffle=True, num_workers=8)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    fis = {n: torch.zeros_like(p) for n, p in params.items()}
    for b, (x, y, _) in enumerate(dl):
        if b >= n_batches:
            break
        model.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = embed(model, x.cuda())
        F.cross_entropy(logits.float(), y.cuda()).backward()
        for n, p in params.items():
            fis[n] += p.grad.detach() ** 2 / n_batches
    model.zero_grad()
    return fis


# ----------------------------------------------------------------------------- one task

def train_task(model, prev, train_videos, args, mem=None, fisher=None, anchor=None):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    items = frames(train_videos)
    dl = DataLoader(FrameSet(items, model.feature_extractor.preprocess, True), batch_size=args.batch,
                    shuffle=True, drop_last=True, num_workers=8)
    total = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))
    m = args.method
    z_old_cache = None
    if mem is not None:  # features of the training frames under the model at task start (for drift)
        z_old_cache, y_cache, _ = extract(prev, train_videos, per_video=None)
    log = []
    for ep in range(args.epochs):
        model.train()
        t0, agg = time.time(), {}
        for x, y, _ in dl:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, z = embed(model, x)
                if m in ("lwf", "featkd", "radial"):
                    with torch.no_grad():
                        logits_p, z_p = embed(prev, x)
            loss = gend_loss(logits, z, y)
            terms = {"gend": loss.item()}
            if m == "lwf":
                T = 2.0
                kd = F.kl_div(F.log_softmax(logits.float() / T, 1), F.softmax(logits_p.float() / T, 1), reduction="batchmean") * T * T
                loss = loss + args.lam * kd
                terms["kd"] = kd.item()
            if m == "featkd":
                kd = (1 - (z * z_p.float()).sum(1)).mean()
                loss = loss + args.lam * kd
                terms["kd"] = kd.item()
            if m == "radial" and (y == 0).any():
                kd = (1 - (z[y == 0] * z_p.float()[y == 0]).sum(1)).mean()
                loss = loss + args.lam * kd
                terms["kd"] = kd.item()
            if m == "ewc":
                pen = sum((fisher[n] * (p - anchor[n]) ** 2).sum() for n, p in model.named_parameters() if p.requires_grad)
                loss = loss + args.lam * pen
                terms["ewc"] = pen.item()
            if mem is not None:
                zr, yr = replay_batch(mem, args.replay)
                rep = F.cross_entropy(model.model(zr).float(), yr)  # head only: replayed features are fixed vectors
                loss = loss + args.lam_rep * rep
                terms["rep"] = rep.item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            for k, v in terms.items():
                agg[k] = agg.get(k, 0) + v / len(dl)
        if mem is not None:  # update prototype drift with the current model
            z_new, _, _ = extract(model, train_videos, per_video=None)
            compensate(mem, z_old_cache, z_new, y_cache, real_anchor=(m == "radial"))
            z_old_cache = z_new
        log.append({"epoch": ep, "sec": round(time.time() - t0, 1), **{k: round(v, 5) for k, v in agg.items()}})
        print(json.dumps(log[-1]), flush=True)
    return log


# ----------------------------------------------------------------------------- stream

def evaluate_all(model, reg, stream):
    return {t: score(model, videos(reg, t, "dev")) for t in stream + ["FSh_probe"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["seq", "lwf", "ewc", "featkd", "lnmerge", "sdc", "radial"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stream", default="FF,DFDCP,DFD,CDF")
    ap.add_argument("--init", default="yermandy/GenD_CLIP_L_14")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--lam_rep", type=float, default=1.0)
    ap.add_argument("--replay", type=int, default=64)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--tag", default="dev")
    args = ap.parse_args()

    # share the GPU with the GenD retrain safely: an OOM here must never hit the other process
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("RADIAL_GPU_FRACTION", "0.42")))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    reg = load_registration()
    stream = args.stream.split(",")
    run = f"{OUT}/{args.tag}/{args.method}_seed{args.seed}"
    os.makedirs(run, exist_ok=True)
    json.dump(vars(args) | {"splits": reg["version"]}, open(f"{run}/args.json", "w"), indent=1)

    model = load_model(args.init)
    results = {"stages": [], "args": vars(args)}

    # base task (already learned by the checkpoint): memory / Fisher from a fixed FF fit subset
    base_videos = few_shot(reg, stream[0], seed=1000 + args.seed, n_real=100, n_fake=100)
    mem = fisher = anchor = None
    if args.method in ("sdc", "radial"):
        z, y, _ = extract(model, base_videos)
        mem = build_memory(z, y, args.k)
    if args.method == "ewc":
        fisher, anchor = fisher_diag(model, base_videos), {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    merged = trainable_state(model) if args.method == "lnmerge" else None

    results["stages"].append({"after": stream[0], "dev": evaluate_all(model, reg, stream)})
    print(json.dumps({"after": stream[0], "dev": {t: round(v["auroc"], 4) for t, v in results["stages"][-1]["dev"].items()}}), flush=True)

    for i, task in enumerate(stream[1:], start=1):
        train_videos = few_shot(reg, task, args.seed)
        prev = copy.deepcopy(model).eval()
        for p in prev.parameters():
            p.requires_grad = False
        prev.to(torch.bfloat16)
        log = train_task(model, prev, train_videos, args, mem=mem, fisher=fisher, anchor=anchor)
        if args.method == "lnmerge":  # running average over all tasks learned so far (incl. base)
            cur = trainable_state(model)
            merged = {n: (merged[n] * i + cur[n]) / (i + 1) for n in merged}
            model.load_state_dict({n: v.cuda() for n, v in merged.items()}, strict=False)
        if args.method in ("sdc", "radial"):
            z, y, _ = extract(model, train_videos)
            mem += build_memory(z, y, args.k)
        if args.method == "ewc":
            f_new = fisher_diag(model, train_videos)
            fisher = {n: fisher[n] + f_new[n] for n in fisher}
            anchor = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        del prev
        torch.cuda.empty_cache()
        results["stages"].append({"after": task, "train_log": log, "n_train_videos": len(train_videos),
                                  "dev": evaluate_all(model, reg, stream)})
        print(json.dumps({"after": task, "dev": {t: round(v["auroc"], 4) for t, v in results["stages"][-1]["dev"].items()}}), flush=True)
        torch.save(trainable_state(model), f"{run}/after_{task}.pt")
        json.dump(results, open(f"{run}/results.json", "w"), indent=1)

    if mem is not None:
        torch.save([{k: (v.cpu() if torch.is_tensor(v) else v) for k, v in m.items()} for m in mem], f"{run}/memory.pt")
    json.dump(results, open(f"{run}/results.json", "w"), indent=1)


if __name__ == "__main__":
    main()

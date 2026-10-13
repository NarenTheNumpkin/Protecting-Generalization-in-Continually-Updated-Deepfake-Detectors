"""Continual adaptation of GenD over a task stream (default S1: FF -> DFDCP -> DFD -> CDF).

Methods (all tune only LayerNorm affine params + the LinearNorm head, starting from a GenD checkpoint
that has already learned FF):
  seq     naive sequential fine-tuning with the GenD loss (lower bound; Gate 1)
  lwf     + logit distillation from the previous model on current data (Li & Hoiem)
  ewc     + diagonal-Fisher quadratic penalty, Fisher estimated at the end of each task (Kirkpatrick et al.)
  featkd  + symmetric feature distillation 1-cos(z, z_prev) on ALL current samples (control for radial)
  lnmerge seq per task, then running average of LN/head params across tasks (model-merging control)
  sdc     + vMF prototype replay for the head with SDC drift compensation from ALL current samples
  realkd  real-only feature distillation, no prototype replay (exploratory E1, post Gate 2)
  realkd_lwf real-only feature distillation + LwF, no prototype replay (exploratory E2)
  radial_lwf radial + LwF logit distillation, same lambda (exploratory E1, post Gate 2)
  --- journal extension J1 (registration/journal_J1.md) ---
  featkd_lwf symmetric feature distillation + LwF (all-sample distillation; ablation of realkd_lwf)
  fakekd  fake-only feature distillation (mirror control of realkd)
  randkd  feature distillation on a random subset of the batch, as many samples as it has genuine ones (matched-count control)
  head    head-only fine-tuning (LayerNorm frozen)
  replay  exemplar replay: a small buffer of raw videos from every earlier task (stores data; reference)
  joint   oracle upper bound: one training run on the union of all few-shot sets (not sequential)
  --- journal extension J3 (registration/journal_J3.md) ---
  gpl     GPL (Zhang et al., ICCV 2025) re-implemented on GenD: frozen detector (LayerNorm + head frozen), a learnable
          input watermark generator, hyperbolic contrastive alignment to base-task cluster centres and gradient
          projection against the base-aligned KL gradient. Only the watermark generator is trained.
  Long streams: a task token "DFDCP#2" is the 2nd of disjoint few-shot sessions drawn from the DFDCP fitting pool.
  radial  + asymmetric REAL-ONLY feature distillation + vMF prototype replay whose drift is estimated
          from current genuine frames for real prototypes (all frames for fake prototypes)
Memory for prototype methods: K spherical k-means centroids + vMF concentration per (task, class); no images.

Evaluation after every stage: dev split of every task + the FaceShifter probe (never trained),
video-level AUROC and ACC@0.5 (8 evenly spaced frames per video on dev).
"""

import paths
import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import FrameSet, embed, few_shot, frames, gend_loss, load_model, load_registration, score, score_panel, trainable_state, videos

OUT = paths.RUNS


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


# ----------------------------------------------------------------------------- GPL (re-implementation)

class WatermarkGenerator(torch.nn.Module):
    """GPL supplement, App. A: three 3x3 conv layers (3->8->16->32), each followed by 2x2 max pooling, then a
    three-channel mask at 1/8 resolution, nearest-neighbour upsampled to the input size (Eq. 8) and multiplied by a
    sample-shared watermark theta_W (Eq. 9): x_w = x + lam_w * attn(x) * theta_W. theta_W starts at zero, so the
    initial model is exactly the base detector (the GenD base task was trained without a watermark)."""

    def __init__(self, size, lam_w=1.0):
        super().__init__()
        c = [3, 8, 16, 32]
        layers = []
        for a, b in zip(c[:-1], c[1:]):
            layers += [torch.nn.Conv2d(a, b, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.MaxPool2d(2)]
        layers += [torch.nn.Conv2d(32, 3, 1), torch.nn.Sigmoid()]  # attention mask in [0, 1]
        self.conv = torch.nn.Sequential(*layers)
        self.theta_w = torch.nn.Parameter(torch.zeros(3, size[0], size[1]))
        self.lam_w = lam_w

    def forward(self, x):
        attn = F.interpolate(self.conv(x), size=x.shape[-2:], mode="nearest")
        return x + self.lam_w * attn * self.theta_w


class WMExtractor(torch.nn.Module):
    """Feature extractor with the GPL watermark applied to its (preprocessed) input."""

    def __init__(self, fe, wm):
        super().__init__()
        self.fe, self.wm = fe, wm

    @property
    def preprocess(self):
        return self.fe.preprocess

    def forward(self, x):
        return self.fe(self.wm(x))


def poincare(z, c=1.0):
    """Exponential map at the origin of the Poincare ball (curvature -c). GenD features are unit-norm, so every
    feature lands on the sphere of radius tanh(sqrt(c)) inside the ball."""
    n = z.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return torch.tanh(math.sqrt(c) * n) * z / (math.sqrt(c) * n)


def hyp_dist(a, b):
    """GPL Eq. 10, pairwise: a [N, d], b [M, d] -> [N, M]."""
    d2 = torch.cdist(a, b) ** 2
    den = (1 - (a ** 2).sum(1, keepdim=True)) * (1 - (b ** 2).sum(1)).unsqueeze(0)
    return torch.acosh((1 + 2 * d2 / den.clamp(min=1e-6)).clamp(min=1 + 1e-6))


def gpl_nce(z, y, centers, tau):
    """GPL Eq. 11: pull towards the nearest same-class base centre, push from all opposite-class centres."""
    h = poincare(z.float())
    loss = 0.0
    for lab in (0, 1):
        sel = y == lab
        if not sel.any():
            continue
        dp = hyp_dist(h[sel], poincare(centers[lab])).min(1).values  # nearest same-class centre
        dn = hyp_dist(h[sel], poincare(centers[1 - lab]))  # all opposite-class centres
        logits = torch.cat([-dp.unsqueeze(1), -dn], 1) / tau
        loss = loss + F.cross_entropy(logits, torch.zeros(int(sel.sum()), dtype=torch.long, device=z.device), reduction="sum")
    return loss / len(y)


def setup_gpl(model, args):
    x = model.feature_extractor.preprocess(__import__("PIL.Image", fromlist=["Image"]).new("RGB", (256, 256)))
    wm = WatermarkGenerator(tuple(x.shape[-2:]), args.gpl_lam_w).cuda()
    model.feature_extractor = WMExtractor(model.feature_extractor, wm)
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("feature_extractor.wm.")  # backbone and classifier frozen (GPL Sec. 5.1)
    return model


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

def train_task(model, prev, train_videos, args, mem=None, fisher=None, anchor=None, centers=None):
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
    torch.cuda.reset_peak_memory_stats()
    t_task = time.time()
    for ep in range(args.epochs):
        model.train()
        t0, agg = time.time(), {}
        for x, y, _ in dl:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, z = embed(model, x)
                if m in ("lwf", "featkd", "radial", "realkd", "radial_lwf", "realkd_lwf", "featkd_lwf", "fakekd", "randkd", "gpl"):
                    with torch.no_grad():
                        logits_p, z_p = embed(prev, x)
            if m == "gpl":  # GPL Alg. 1: L_total = L_cls + lam_nce L_nce; G_align = G_task - lam_g (G_task.G_base)/|G_base|^2 G_base if conflicting
                l_cls = F.cross_entropy(logits.float(), y)
                l_nce = gpl_nce(z, y, centers, args.gpl_tau)
                l_tot = l_cls + args.gpl_lam_nce * l_nce
                kl = F.kl_div(F.log_softmax(logits.float(), 1), F.softmax(logits_p.float(), 1), reduction="batchmean")  # KL(p_fix || p_learn)
                g_task = torch.autograd.grad(l_tot, params, retain_graph=True)
                g_base = torch.autograd.grad(kl, params)
                gt = torch.cat([g.flatten() for g in g_task]); gb = torch.cat([g.flatten() for g in g_base])
                dot, nb = float(gt @ gb), float(gb @ gb)
                if dot < 0 and nb > 1e-20:  # angle > 90 degrees: remove the conflicting component
                    gt = gt - args.gpl_lam_g * dot / nb * gb
                k = 0
                for p in params:
                    p.grad = gt[k:k + p.numel()].view_as(p).to(p.dtype)
                    k += p.numel()
                opt.step()
                sched.step()
                for kk, v in {"cls": l_cls.item(), "nce": l_nce.item(), "kl": kl.item(), "projected": float(dot < 0)}.items():
                    agg[kk] = agg.get(kk, 0) + v / len(dl)
                continue
            loss = gend_loss(logits, z, y)
            terms = {"gend": loss.item()}
            if m in ("lwf", "radial_lwf", "realkd_lwf", "featkd_lwf"):
                T = 2.0
                kd = F.kl_div(F.log_softmax(logits.float() / T, 1), F.softmax(logits_p.float() / T, 1), reduction="batchmean") * T * T
                loss = loss + (args.lam if args.lam_lwf is None else args.lam_lwf) * kd
                terms["kd_lwf"] = kd.item()
            if m in ("featkd", "featkd_lwf"):
                kd = (1 - (z * z_p.float()).sum(1)).mean()
                loss = loss + args.lam * kd
                terms["kd"] = kd.item()
            if m in ("radial", "realkd", "radial_lwf", "realkd_lwf") and (y == 0).any():
                kd = (1 - (z[y == 0] * z_p.float()[y == 0]).sum(1)).mean()
                loss = loss + args.lam * kd
                terms["kd"] = kd.item()
            if m == "fakekd" and (y == 1).any():
                kd = (1 - (z[y == 1] * z_p.float()[y == 1]).sum(1)).mean()
                loss = loss + args.lam * kd
                terms["kd"] = kd.item()
            if m == "randkd" and (y == 0).any():  # as many distilled samples as GAD, chosen regardless of label
                sel = torch.randperm(len(y), device=y.device)[: int((y == 0).sum())]
                kd = (1 - (z[sel] * z_p.float()[sel]).sum(1)).mean()
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
            compensate(mem, z_old_cache, z_new, y_cache, real_anchor=(m in ("radial", "radial_lwf")))
            z_old_cache = z_new
        log.append({"epoch": ep, "sec": round(time.time() - t0, 1), **{k: round(v, 5) for k, v in agg.items()}})
        print(json.dumps(log[-1]), flush=True)
    log.append({"task_train_sec": round(time.time() - t_task, 1), "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                "n_trainable": int(sum(p.numel() for p in params)), "n_frames": len(items)})
    print(json.dumps(log[-1]), flush=True)
    return log


# ----------------------------------------------------------------------------- stream

def base(t):
    return t.split("#")[0]


def unique_tasks(stream):
    return list(dict.fromkeys(base(t) for t in stream))


def session_shot(reg, token, seed, n):
    """Few-shot set of a stream token. "DFDCP#k": k-th of disjoint n+n sessions from a seeded shuffle of the pool."""
    if "#" not in token:
        return few_shot(reg, token, seed, n_real=n, n_fake=n)
    task, k = token.split("#")
    rng = random.Random(f"{task}-{seed}-sessions")
    out = []
    for lab in (0, 1):
        pool = sorted(videos(reg, task, "fit", lab))
        rng.shuffle(pool)
        part = pool[(int(k) - 1) * n: int(k) * n]
        assert len(part) == n, (token, lab, len(pool))
        out += part
    return out


def evaluate_all(model, reg, stream, role="dev", tasks=None):
    out = {t: score(model, videos(reg, t, role)) for t in (unique_tasks(stream) if tasks is None else list(dict.fromkeys(base(t) for t in tasks)))}
    out["FSh_probe"] = score(model, videos(reg, "FSh_probe", "dev"))  # FSh probe only exists as dev
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["none", "seq", "lwf", "ewc", "featkd", "lnmerge", "sdc", "radial", "realkd", "radial_lwf",
                                                             "realkd_lwf", "featkd_lwf", "fakekd", "randkd", "head", "replay", "joint", "gpl"])
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
    ap.add_argument("--eval-role", default="dev", choices=["dev", "test"])  # C1: sealed test splits
    ap.add_argument("--panel", action="store_true")  # C1: score the sealed unseen panel after the last stage
    ap.add_argument("--lam-lwf", type=float, default=None)  # J1: separate LwF weight (None = --lam, as in all earlier runs)
    ap.add_argument("--shots", type=int, default=25)  # J1: few-shot videos per class per task (earlier runs: 25)
    ap.add_argument("--mem-videos", type=int, default=5)  # J1 replay: raw videos per class kept from every earlier task
    ap.add_argument("--eval-light", action="store_true")  # J1 sweeps: intermediate stages score only the tasks needed for gain
    ap.add_argument("--run-name", default=None)  # J1: run directory name (default <method>_seed<seed>)
    ap.add_argument("--gpl-lam-nce", type=float, default=0.1)  # J3 GPL: best value of the GPL supplement (Tab. 2)
    ap.add_argument("--gpl-tau", type=float, default=0.1)  # J3 GPL: not reported by GPL; fixed in registration
    ap.add_argument("--gpl-lam-g", type=float, default=1.0)  # J3 GPL: projection intensity (1 = remove the conflicting component)
    ap.add_argument("--gpl-lam-w", type=float, default=1.0)  # J3 GPL: watermark scale (theta_W is learned, so only a parametrisation)
    ap.add_argument("--gpl-k", type=int, default=25)  # J3 GPL: N_c = 25 base cluster centres per class
    args = ap.parse_args()

    # share the GPU with the GenD retrain safely: an OOM here must never hit the other process
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("RADIAL_GPU_FRACTION", "0.42")))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    reg = load_registration()
    stream = args.stream.split(",")
    run = f"{OUT}/{args.tag}/{args.run_name or f'{args.method}_seed{args.seed}'}"
    os.makedirs(run, exist_ok=True)
    json.dump(vars(args) | {"splits": reg["version"]}, open(f"{run}/args.json", "w"), indent=1)

    model = load_model(args.init)
    if args.method == "head":
        for n, p in model.named_parameters():
            p.requires_grad = n.startswith("model.")
    if args.method == "gpl":
        model = setup_gpl(model, args)
    results = {"stages": [], "args": vars(args)}

    # base task (already learned by the checkpoint): memory / Fisher from a fixed FF fit subset
    base_videos = few_shot(reg, stream[0], seed=1000 + args.seed, n_real=100, n_fake=100)
    mem = fisher = anchor = centers = None
    if args.method == "gpl":  # base-task cluster centres of genuine and manipulated features (GPL Sec. 4.1)
        z, y, _ = extract(model, base_videos)
        centers = {lab: spherical_kmeans(z[y == lab], args.gpl_k, seed=args.seed)[0] for lab in (0, 1)}
    if args.method in ("sdc", "radial", "radial_lwf"):
        z, y, _ = extract(model, base_videos)
        mem = build_memory(z, y, args.k)
    if args.method == "ewc":
        fisher, anchor = fisher_diag(model, base_videos), {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    merged = trainable_state(model) if args.method == "lnmerge" else None

    base_tasks = [stream[1]] if (args.eval_light and args.method not in ("none", "joint")) else None
    results["stages"].append({"after": stream[0], "dev": evaluate_all(model, reg, stream, args.eval_role, base_tasks)})
    if args.method == "joint":  # oracle: one run on the union of all few-shot sets (FF sample + every incremental task)
        union = few_shot(reg, stream[0], seed=3000 + args.seed, n_real=args.shots, n_fake=args.shots)
        for t in stream[1:]:
            union += few_shot(reg, t, args.seed, n_real=args.shots, n_fake=args.shots)
        log = train_task(model, None, union, args)
        results["stages"].append({"after": "JOINT", "train_log": log, "n_train_videos": len(union),
                                  "dev": evaluate_all(model, reg, stream, args.eval_role)})
        torch.save(trainable_state(model), f"{run}/after_JOINT.pt")
        if args.panel:
            results["panel"], preds = score_panel(model)
            json.dump(preds, open(f"{run}/panel_preds.json", "w"))
        json.dump(results, open(f"{run}/results.json", "w"), indent=1)
        print(json.dumps({"after": "JOINT", "dev": {t: round(v["auroc"], 4) for t, v in results["stages"][-1]["dev"].items()}}), flush=True)
        return
    buffer = few_shot(reg, stream[0], seed=2000 + args.seed, n_real=args.mem_videos, n_fake=args.mem_videos) if args.method == "replay" else []
    if args.method == "none":  # no-update reference: base stage only (+ panel)
        if args.panel:
            results["panel"], preds = score_panel(model)
            json.dump(preds, open(f"{run}/panel_preds.json", "w"))
        json.dump(results, open(f"{run}/results.json", "w"), indent=1)
        print(json.dumps({"panel": {k: round(v["auroc"], 4) for k, v in results.get("panel", {}).items()}}), flush=True)
        return
    print(json.dumps({"after": stream[0], "dev": {t: round(v["auroc"], 4) for t, v in results["stages"][-1]["dev"].items()}}), flush=True)

    for i, task in enumerate(stream[1:], start=1):
        train_videos = session_shot(reg, task, args.seed, args.shots)
        prev = copy.deepcopy(model).eval()
        for p in prev.parameters():
            p.requires_grad = False
        prev.to(torch.bfloat16)
        n_used = len(train_videos) + len(buffer)  # videos actually trained on in this task (buffer before this task is added)
        log = train_task(model, prev, train_videos + buffer, args, mem=mem, fisher=fisher, anchor=anchor, centers=centers)
        if args.method == "replay":  # keep mem_videos genuine + mem_videos fake raw videos of this task
            rr = random.Random(f"replay-{task}-{args.seed}")
            buffer += rr.sample([v for v in train_videos if v[1] == 0], args.mem_videos) + rr.sample([v for v in train_videos if v[1] == 1], args.mem_videos)
        if args.method == "lnmerge":  # running average over all tasks learned so far (incl. base)
            cur = trainable_state(model)
            merged = {n: (merged[n] * i + cur[n]) / (i + 1) for n in merged}
            model.load_state_dict({n: v.cuda() for n, v in merged.items()}, strict=False)
        if args.method in ("sdc", "radial", "radial_lwf"):
            z, y, _ = extract(model, train_videos)
            mem += build_memory(z, y, args.k)
        if args.method == "ewc":
            f_new = fisher_diag(model, train_videos)
            fisher = {n: fisher[n] + f_new[n] for n in fisher}
            anchor = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        del prev
        torch.cuda.empty_cache()
        last = i == len(stream) - 1
        stage_tasks = None if (last or not args.eval_light) else [base(task), base(stream[i + 1])]
        results["stages"].append({"after": task, "train_log": log, "n_train_videos": n_used,
                                  "dev": evaluate_all(model, reg, stream, args.eval_role, stage_tasks)})
        print(json.dumps({"after": task, "dev": {t: round(v["auroc"], 4) for t, v in results["stages"][-1]["dev"].items()}}), flush=True)
        torch.save(trainable_state(model), f"{run}/after_{task}.pt")
        json.dump(results, open(f"{run}/results.json", "w"), indent=1)

    if args.panel:
        results["panel"], preds = score_panel(model)
        json.dump(preds, open(f"{run}/panel_preds.json", "w"))
        print(json.dumps({"panel": {k: round(v["auroc"], 4) for k, v in results["panel"].items()}}), flush=True)
    if mem is not None:
        torch.save([{k: (v.cpu() if torch.is_tensor(v) else v) for k, v in m.items()} for m in mem], f"{run}/memory.pt")
    json.dump(results, open(f"{run}/results.json", "w"), indent=1)


if __name__ == "__main__":
    main()

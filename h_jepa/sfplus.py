# ScheduleFree+ (AdamC + Schedule-Free + Polyak step size), Defazio, arXiv 2605.19095 (May 2026).
# A single-process port of facebookresearch/schedule_free schedulefree/adamc_schedulefree_plus_paper.py
# (Copyright (c) Meta Platforms, Inc. and affiliates; Apache-2.0). Changes: no torch.distributed /
# DTensor paths; step() takes the loss from `function_value`, which the training module sets before
# each step (stable-pretraining steps without a closure); the update rule is unchanged.
import math
import os

import torch


class AdamCScheduleFreePlus(torch.optim.Optimizer):
    """No learning rate and no schedule: the step size is a Polyak step, max(0, f + beta <g, z - x>)
    / EMA(|g|_1 sqrt(pi/2)), with f the training loss (f* taken as 0, as in the paper's Algorithm
    1); the schedule is Schedule-Free averaging (x: evaluation weights, y: gradient point, z: AdamW
    iterate). Weight decay is AdamC: decoupled, scaled by lr^2, so its values are like 5 to 50.
    Call .train() / .eval() as for every Schedule-Free optimizer (h_jepa ScheduleFreeModes)."""

    def __init__(self, params, lr=1.0, betas=(0.9, 0.95), sf_beta1=0.9, eps=1e-8, weight_decay=0.0, r=0.0,
                 polyak_beta=0.0, c_warmup=0, sf_beta1_anneal_steps=0, sf_beta1_max=0.965, weight_lr_power=2.0):
        defaults = dict(lr=lr, betas=tuple(betas), sf_beta1=sf_beta1, eps=eps, r=r, k=0, train_mode=False,
                        weight_sum=0.0, lr_max=eps, scheduled_lr=0.0, polyak_beta=polyak_beta,
                        sf_beta1_anneal_steps=sf_beta1_anneal_steps, sf_beta1_max=sf_beta1_max,
                        grad_l1_ema=0.0, c_warmup=c_warmup, weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.function_value = None

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            if group["train_mode"]:
                for p in group["params"]:
                    if "x" in self.state[p]:
                        p.copy_(self.state[p]["x"])
                group["train_mode"] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            if not group["train_mode"]:
                for p in group["params"]:
                    if "y" in self.state[p]:
                        p.copy_(self.state[p]["y"])
                group["train_mode"] = True

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:  # Lightning's manual optimization passes a closure that returns None
            with torch.enable_grad():
                loss = closure()
            if loss is not None:
                self.function_value = float(loss)
        if self.function_value is None:
            raise RuntimeError("AdamCScheduleFreePlus needs the loss: set .function_value before step()")
        g0 = self.param_groups[0]
        if not g0["train_mode"]:
            raise RuntimeError("AdamCScheduleFreePlus.step() outside train mode: call .train() first")
        k, pb = g0["k"], g0["polyak_beta"]
        if g0["sf_beta1_anneal_steps"] > 0:
            prog = min(k / g0["sf_beta1_anneal_steps"], 1.0)
            sf_beta1 = 1 - math.exp(math.log(1 - g0["sf_beta1"]) * (1 - prog) + math.log(1 - g0["sf_beta1_max"]) * prog)
        else:
            sf_beta1 = g0["sf_beta1"]

        l1, ip = [], []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                l1.append(torch.linalg.vector_norm(p.grad, ord=1))
                st = self.state[p]
                if "z" in st:
                    ip.append(sf_beta1 * (p.grad * (st["z"] - st["x"])).sum())
        grad_l1 = torch.stack(l1).sum().item() if l1 else 0.0
        ip_term = torch.stack(ip).sum().item() if ip else 0.0
        ema = pb * g0["grad_l1_ema"] + (1 - pb) * grad_l1 * math.sqrt(math.pi / 2)
        polyak_lr = max(0.0, self.function_value + ip_term) / max(ema / (1 - pb ** (k + 1)), 1e-30)
        if os.environ.get("SFPLUS_DEBUG") and (k % 10 == 0 or os.environ["SFPLUS_DEBUG"] == "all"):
            print(f"sfplus k={k} f={self.function_value:.4g} ip={ip_term:.4g} l1={grad_l1:.4g} polyak_lr={polyak_lr:.4g} "
                  f"warm={g0['lr']:.3g}", flush=True)

        for group in self.param_groups:
            group["grad_l1_ema"] = ema
            eps, decay, (beta1, beta2) = group["eps"], group["weight_decay"], group["betas"]
            k = group["k"]
            group_lr = max(group["lr"], eps) * polyak_lr  # group['lr'] carries any warmup factor
            group["scheduled_lr"] = group_lr
            lr_max = group["lr_max"] = max(group_lr, group["lr_max"])
            if k < group["c_warmup"]:
                ckp1 = 1.0
            else:
                weight = ((k + 1) ** group["r"]) * (lr_max ** group["weight_lr_power"])
                group["weight_sum"] += weight
                ckp1 = weight / group["weight_sum"]
            bc1, bc2 = 1 - beta1 ** (k + 1), 1 - beta2 ** (k + 1)
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "z" not in st:
                    st["z"], st["x"], st["y"] = p.detach().clone(), p.detach().clone(), p.detach().clone()
                    st["exp_avg"], st["exp_avg_sq"] = torch.zeros_like(p), torch.zeros_like(p)
                z, x, y, m, v = st["z"], st["x"], st["y"], st["exp_avg"], st["exp_avg_sq"]
                z.sub_(y, alpha=group_lr * group_lr * decay)  # AdamC decoupled weight decay at y
                m.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                z.addcdiv_(m / bc1, (v / bc2).sqrt_().add_(eps), value=-group_lr)
                x.mul_(1 - ckp1).add_(z, alpha=ckp1)
                y.copy_(x.mul(sf_beta1).add_(z, alpha=1 - sf_beta1))
                p.copy_(y)
            group["k"] = k + 1
        return self.function_value

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_module_weights(m, std=0.02):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (ECF means averaged across DDP ranks)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        cos, sin = x_t.cos().mean(-3), x_t.sin().mean(-3)
        n = proj.size(-2)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # untracked by autograd: each rank backprops its local ECF, DDP's grad average
            # then yields the gradient of the global statistic
            world_size = torch.distributed.get_world_size()
            for ecf in (cos, sin):
                torch.distributed.all_reduce(ecf)
                ecf.data.div_(world_size)
            n = n * world_size
        err = (cos - self.phi).square() + sin.square()
        statistic = (err @ self.weights) * n
        return statistic.mean()


class RDMReg(torch.nn.Module):
    """RDMReg of LpWM (arXiv 2608.22764; github.com/YilunKuang/lpworldmodel, lpwm_swm/loss.py):
    a sliced-Wasserstein match of the embeddings to a rectified generalized Gaussian. p=1 with the
    ReLU link is a rectified product Laplace: non-negative, sparse codes. p=2 with the Identity
    link is the dense isotropic Gaussian (a LeWM-like control).

    Same call as SIGReg: proj (T, B, D). matching_mode 'b_t_d' (the LpWM default) matches each
    time step over the batch; 'bt_d' pools all frames. rms_norm scales the ReLU target to unit
    second moment (the paper's sparse runs). Location mu is 0.
    """

    def __init__(self, p=1.0, link="ReLU", num_slices=1024, matching_mode="b_t_d", rms_norm=True):
        super().__init__()
        if link not in ("ReLU", "Identity") or matching_mode not in ("b_t_d", "bt_d"):
            raise ValueError(f"RDMReg: unsupported link={link!r} or matching_mode={matching_mode!r}")
        self.p, self.link = float(p), link
        self.num_slices, self.matching_mode = int(num_slices), matching_mode
        # sigma gives the generalized Gaussian unit variance; then E[ReLU(X)^2] = 1/2.
        self.sigma = math.sqrt(math.gamma(1 / self.p) / math.gamma(3 / self.p)) / self.p ** (1 / self.p)
        self.scale = math.sqrt(2.0) if rms_norm and link == "ReLU" else 1.0

    def sample(self, shape, device):
        if self.p == 1.0:  # Laplace by the inverse CDF, on the device
            u = (torch.rand(shape, device=device) - 0.5).clamp(-0.5 + 1e-7, 0.5 - 1e-7)
            x = -self.sigma * torch.sign(u) * torch.log1p(-2 * u.abs())
        elif self.p == 2.0:
            x = self.sigma * torch.randn(shape, device=device)
        else:
            sign = torch.randint(0, 2, shape, device=device) * 2 - 1
            g = torch.distributions.Gamma(torch.tensor(1 / self.p, device=device), 1.0).sample(shape)
            x = self.sigma * sign * (self.p * g).pow(1 / self.p)
        if self.link == "ReLU":
            x = F.relu(x)
        return x * self.scale

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        z = proj.float()
        if self.matching_mode == "bt_d":
            z = z.reshape(1, -1, z.size(-1))
        dirs = torch.randn(z.size(-1), self.num_slices, device=z.device)
        dirs = dirs / dirs.norm(dim=0, keepdim=True)
        a = torch.sort(z @ dirs, dim=1).values
        b = torch.sort(self.sample(tuple(z.shape), z.device) @ dirs, dim=1).values
        return (a - b).square().mean()


class InverseDynamicsModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(init_module_weights)

    def forward(self, state_t, state_t_plus_1):
        return self.model(torch.cat([state_t.flatten(1), state_t_plus_1.flatten(1)], dim=1))


class EndpointInverseDynamicsLoss(nn.Module):
    """Endpoint inverse dynamics (EP-IDM; Toso, LeCun, Anderson, Bounou, arXiv 2610.07540): an MLP
    reconstructs the whole action sequence a_s..a_{s+H-1} from only the latents z_s and z_{s+H}.
    Exact reconstruction keeps every state direction that actions reach within H steps, so the encoder
    cannot drop what the actions change (e.g. a cube that the arm moves). Every start s of the clip
    with s + H inside it is used.
    """

    def __init__(self, state_dim, hidden_dim, action_dim, horizon):
        super().__init__()
        self.horizon, self.action_dim = int(horizon), int(action_dim)
        self.model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.horizon * self.action_dim),
        )
        self.apply(init_module_weights)

    def forward(self, emb, action):
        """
        emb: (B, T, D) with T >= horizon + 1, action: (B, T, A) or (B, T-1, A)
        """
        H, starts = self.horizon, emb.size(1) - self.horizon
        if starts < 1:
            raise ValueError(f"EP-IDM needs clips of horizon + 1 = {H + 1} frames, got {emb.size(1)}")
        z0 = emb[:, :starts].flatten(0, 1)
        zH = emb[:, H : H + starts].flatten(0, 1)
        pred = self.model(torch.cat([z0.flatten(1), zH.flatten(1)], dim=1)).view(-1, H, self.action_dim)
        target = torch.stack([action[:, s : s + H] for s in range(starts)], dim=1).flatten(0, 1)
        return F.mse_loss(pred, target.detach())


class InverseDynamicsLoss(nn.Module):
    """MSE between the IDM action predicted from (z_t, z_t+1) and the raw action a_t."""

    def __init__(self, idm):
        super().__init__()
        self.idm = idm

    def forward(self, emb, action):
        """
        emb: (B, T, D), action: (B, T, A) or (B, T-1, A)
        """
        pred = self.idm(emb[:, :-1].flatten(0, 1), emb[:, 1:].flatten(0, 1))  # [B*(T-1), A]
        target = action[:, : emb.size(1) - 1].flatten(0, 1)  # [B*(T-1), A]
        return F.mse_loss(pred, target.detach())

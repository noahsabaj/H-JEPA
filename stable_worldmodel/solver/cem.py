"""Cross-entropy method (CEM) solver: the planner of LeWM and LpWM."""

import time

import torch

from .gd import GradientSolver
from .solver import get_planning_horizon


class CEMSolver(GradientSolver):
    """Cross-entropy method over action sequences, with the GradientSolver interface.

    Each iteration samples num_samples sequences from a diagonal Gaussian (the first sample is its
    mean), scores them with the world model's cost (no gradients), and refits the Gaussian to the
    topk lowest-cost sequences. The Gaussian starts at the action prior (scaled by var_scale) and,
    when given, at the warm-start actions of the previous plan. The final mean is the plan.

    Defaults are LpWM's planner (lpwm_swm/config/plan/solver/cem.yaml): 300 samples, top 30,
    30 iterations, var_scale 1.

    Args:
        model: World model implementing the Costable protocol.
        n_steps: CEM iterations.
        num_samples: Sequences sampled per iteration.
        topk: Elite sequences used to refit the Gaussian.
        var_scale: Scale of the prior std.
        batch_size: Number of environments to process in parallel.
        action_clip_sigma: Clip samples to prior mean ± sigma·std; None uses queue quantiles.
        device: Device for tensor computations.
        seed: Random seed for reproducibility.
    """

    MIN_STD = 1e-4  # as the GradientSolver's action prior

    def __init__(
        self,
        model,
        n_steps: int = 30,
        num_samples: int = 300,
        topk: int = 30,
        var_scale: float = 1.0,
        batch_size: int | None = None,
        action_clip_sigma: float | None = None,
        device: str | torch.device = 'cpu',
        seed: int = 1234,
    ) -> None:
        super().__init__(
            model,
            n_steps=n_steps,
            min_its=0,
            batch_size=batch_size,
            var_scale=var_scale,
            num_samples=num_samples,
            early_stop_patience=0,
            action_clip_sigma=action_clip_sigma,
            device=device,
            seed=seed,
        )
        if not 0 < topk <= num_samples:
            raise ValueError(f'topk must be in [1, num_samples]: got {topk} of {num_samples}')
        self.topk = int(topk)

    @torch.inference_mode()
    def solve(
        self,
        info_dict: dict,
        init_action: torch.Tensor | None = None,
        planning_horizon: int | None = None,
        steps_taken: int | None = None,
        eval_budget: int | None = None,
    ) -> dict:
        """Solve the planning problem with the cross-entropy method."""
        start_time = time.time()
        cost_last_n = max(1, int(self._config.cost_last_n))
        intermediate_cost_weight = float(self._config.intermediate_cost_weight)
        if planning_horizon is None:
            planning_horizon = get_planning_horizon(
                configured_horizon=self.horizon,
                replanning_interval=int(self._config.receding_horizon) * int(self._config.action_block),
                steps_taken=steps_taken,
                eval_budget=eval_budget,
            )

        prior_stats = self._get_action_prior_stats()
        if prior_stats is None:
            base_mean = torch.zeros(self.action_dim, device=self.device)
            base_std = torch.ones(self.action_dim, device=self.device)
        else:
            base_mean, base_std = prior_stats
        mean = base_mean.view(1, 1, -1).expand(self.n_envs, planning_horizon, -1).clone()
        std = (base_std * self.var_scale).view(1, 1, -1).expand(self.n_envs, planning_horizon, -1).clone()
        if init_action is not None:
            warm = init_action.to(self.device)[:, :planning_horizon]
            mean[:, : warm.shape[1]] = warm
        clip_bounds = self._get_action_clip_bounds()

        batch_size = self.batch_size if self.batch_size is not None else self.n_envs
        plans, per_env_time = [], []
        for start_idx in range(0, self.n_envs, batch_size):
            end_idx = min(start_idx + batch_size, self.n_envs)
            current_bs = end_idx - start_idx
            batch_t0 = time.time()
            expanded_infos = self._expand_info_dict_for_samples(
                info_dict, start_idx, end_idx, num_samples=self.num_samples
            )
            m, s = mean[start_idx:end_idx], std[start_idx:end_idx]
            rows = torch.arange(current_bs, device=self.device).unsqueeze(1)
            for _ in range(self.n_steps):
                noise = torch.randn(
                    (current_bs, self.num_samples, *m.shape[1:]),
                    generator=self.torch_gen,
                    device=self.device,
                )
                candidates = m.unsqueeze(1) + s.unsqueeze(1) * noise
                candidates[:, 0] = m
                if clip_bounds is not None:
                    lower, upper = clip_bounds
                    candidates = torch.maximum(torch.minimum(candidates, upper), lower)

                current_info = expanded_infos.copy()
                current_info['cost_last_n'] = cost_last_n
                current_info['intermediate_cost_weight'] = intermediate_cost_weight
                costs = self.model.get_cost(current_info, candidates)
                for k in ('embed_0', 'goal_embed_0'):  # encode the observation and goal once per solve
                    if k in current_info:
                        expanded_infos.setdefault(k, current_info[k])
                assert costs.shape == (current_bs, self.num_samples), (
                    f'Cost should be of shape ({current_bs}, {self.num_samples}), got {tuple(costs.shape)}'
                )
                # a non-finite cost ranks last (topk would rank NaN anywhere); none finite is an error
                if not torch.isfinite(costs).any(dim=1).all():
                    raise FloatingPointError('CEM: every candidate has a non-finite cost')
                costs = torch.where(torch.isfinite(costs), costs, torch.inf)

                elite_idx = torch.topk(costs, self.topk, dim=1, largest=False).indices
                elites = candidates[rows, elite_idx]
                # one elite has no sample std (NaN): its std is 0, and every std is at least MIN_STD
                m = elites.mean(dim=1)
                s = elites.std(dim=1, unbiased=self.topk > 1).clamp_min(self.MIN_STD)

            if not torch.isfinite(m).all():
                raise FloatingPointError('CEM: the plan is not finite')
            plans.append(m.detach().cpu())
            per_env_time.extend([time.time() - batch_t0] * current_bs)

        outputs = {'actions': torch.cat(plans, dim=0)}
        outputs['predictions'] = self._rollout_final_actions(info_dict, outputs['actions'])
        end_time = time.time()
        print(f'CEMSolver.solve completed in {end_time - start_time:.4f} seconds.')

        self.solve_records.append({
            'level': int(getattr(self, 'level', 1)),
            'horizon': int(planning_horizon),
            'num_samples': int(self.num_samples),
            'n_iters': [int(self.n_steps)] * self.n_envs,
            'wall_clock': end_time - start_time,
            'wall_clock_per_env': per_env_time,
        })
        return outputs

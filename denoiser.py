import torch
import torch.nn as nn
from model_jit import JiT_models


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            ffn_bitserial=getattr(args, 'ffn_bitserial', True),
            ffn_scales_path=getattr(args, 'ffn_use_kl_scales', ''),
            ffn_int7_weights_path=getattr(args, 'ffn_int7_weights', ''),
            ffn_weight_clip_pct=getattr(args, 'ffn_weight_clip_pct', 0.0),
            ffn_weight_nbit=getattr(args, 'ffn_weight_nbit', 8),
            ffn_act_nbit_eff=getattr(args, 'ffn_act_nbit', 12),
            ffn_msb_samples=getattr(args, 'ffn_msb_samples', 2),
            ffn_lsb_gain_shift=getattr(args, 'ffn_lsb_gain_shift', 2),
            ffn_adc_nbit=getattr(args, 'ffn_adc_nbit', 10),
            ffn_msb_noise_sigma_lsb=getattr(args, 'ffn_msb_noise_sigma_lsb', 2.0),
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.timestep_schedule = getattr(args, 'timestep_schedule', 'linear')
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)
        self._debug_forward_logged = False
        self._debug_schedule_logged = False

    def _make_timesteps_1d(self, device, dtype):
        schedule = self.timestep_schedule
        if schedule == 'log':
            schedule = 'logit_normal'

        if schedule == 'linear':
            # Paper Table 9 uses linear steps in [0, 1]
            t = torch.linspace(0.0, 1.0, steps=self.steps + 1, device=device, dtype=dtype)
            return t

        if schedule == 'logit_normal':
            # Deterministic schedule matching the *training* time distribution:
            # t = sigmoid(N(P_mean, P_std)) via normal-quantiles.
            # We still force exact endpoints {0, 1} to match the ODE loop assumptions.
            u = torch.linspace(0.0, 1.0, steps=self.steps + 1, device=device, dtype=torch.float32)
            eps = 1e-4
            u = u.clamp(eps, 1.0 - eps)
            # Normal inverse CDF using erfinv: icdf(u) = sqrt(2) * erfinv(2u - 1)
            z = torch.erfinv(2.0 * u - 1.0) * (2.0 ** 0.5)
            t = torch.sigmoid(z * float(self.P_std) + float(self.P_mean)).to(dtype=dtype)
            t[0] = torch.tensor(0.0, device=device, dtype=dtype)
            t[-1] = torch.tensor(1.0, device=device, dtype=dtype)
            return t

        raise ValueError(f"Unknown timestep schedule: {self.timestep_schedule}")

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels):
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), labels_dropped)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)

        # Use (steps + 1) points to form `steps` intervals, matching the Heun+Euler loop below.
        timesteps_1d = self._make_timesteps_1d(device=device, dtype=z.dtype)
        timesteps = timesteps_1d.view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if not self._debug_schedule_logged:
            tmin = float(timesteps_1d.min().item())
            tmax = float(timesteps_1d.max().item())
            print(f"DEBUG [timestep_schedule] schedule={self.timestep_schedule} steps={self.steps} t_min={tmin:.6f} t_max={tmax:.6f}")
            self._debug_schedule_logged = True

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        # conditional
        x_cond = self.net(z, t.flatten(), labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        v = v_uncond + cfg_scale_interval * (v_cond - v_uncond)

        if not self._debug_forward_logged:
            def _stats(tensor):
                return (
                    float(tensor.min().item()),
                    float(tensor.max().item()),
                    float(tensor.mean().item()),
                    float(tensor.std().item())
                )

            print("DEBUG [cond x] min/max/mean/std:", *_stats(x_cond))
            print("DEBUG [uncond x] min/max/mean/std:", *_stats(x_uncond))
            print("DEBUG [cond v] min/max/mean/std:", *_stats(v_cond))
            print("DEBUG [uncond v] min/max/mean/std:", *_stats(v_uncond))
            self._debug_forward_logged = True

        return v

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)

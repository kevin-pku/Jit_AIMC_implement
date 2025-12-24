import argparse
import json
import math
import os
from collections import defaultdict
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr
from denoiser import Denoiser


def kl_divergence_scale(tensor: torch.Tensor, num_bins: int = 2048, num_quant_bins: int = 255) -> torch.Tensor:
    arr = tensor.detach().abs().flatten().cpu().numpy()
    if arr.size == 0:
        return torch.tensor(1.0, dtype=torch.float32)
    max_val = float(arr.max())
    if max_val == 0.0:
        return torch.tensor(1.0, dtype=torch.float32)

    hist, bin_edges = np.histogram(arr, bins=num_bins, range=(0.0, max_val))
    hist = hist.astype(np.float64)

    best_kl = None
    best_scale = None

    for i in range(num_quant_bins, num_bins + 1):
        sliced = hist[:i].copy()
        sliced[-1] += hist[i:].sum()
        if sliced.sum() == 0:
            continue

        quant_bins = np.linspace(0, i, num_quant_bins + 1, dtype=int)
        quant_hist = np.zeros(num_quant_bins, dtype=np.float64)
        for j in range(num_quant_bins):
            start, end = quant_bins[j], quant_bins[j + 1]
            quant_hist[j] = sliced[start:end].sum()

        expand_hist = np.zeros_like(sliced, dtype=np.float64)
        for j in range(num_quant_bins):
            start, end = quant_bins[j], quant_bins[j + 1]
            if end > start and quant_hist[j] > 0:
                expand_hist[start:end] = quant_hist[j] / (end - start)

        p = sliced / sliced.sum()
        q = expand_hist / expand_hist.sum() if expand_hist.sum() > 0 else expand_hist

        mask = (p > 0) & (q > 0)
        if not np.any(mask):
            continue
        kl = (p[mask] * np.log(p[mask] / q[mask])).sum()
        if math.isnan(kl):
            continue

        if best_kl is None or kl < best_kl:
            best_kl = kl
            threshold = bin_edges[i]
            best_scale = threshold / 127.0

    if best_scale is None:
        best_scale = max_val / 127.0

    return torch.tensor(float(best_scale), dtype=torch.float32)


def quantize_int8_symmetric(weight: torch.Tensor, clip_percentile: float = 99.8) -> torch.Tensor:
    """Percentile clip then symmetric INT8 quantization ([-128, 127])."""
    qmin, qmax = -128, 127
    w = weight.detach().cpu().float()
    if clip_percentile > 0.0:
        limit = torch.quantile(w.abs(), clip_percentile / 100.0)
        w = torch.clamp(w, -limit, limit)
    max_abs = w.abs().max()
    scale = max_abs / qmax if max_abs > 0 else torch.tensor(1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
    return q


def build_denoiser_from_checkpoint(args, device):
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    ckpt_args = ckpt['args']
    # ensure required attributes exist
    defaults = dict(
        attn_dropout=0.0,
        proj_dropout=0.0,
        label_drop_prob=0.1,
        P_mean=-0.8,
        P_std=0.8,
        noise_scale=1.0,
        t_eps=5e-2,
        sampling_method='heun',
        num_sampling_steps=50,
        timestep_schedule='linear',
        cfg=1.0,
        interval_min=0.1,
        interval_max=1.0,
        ema_decay1=0.9999,
        ema_decay2=0.9996,
        ffn_fake_quant=False,
        ffn_use_kl_scales='',
        ffn_int7_weights='',
    )
    for k, v in defaults.items():
        if not hasattr(ckpt_args, k):
            setattr(ckpt_args, k, v)

    # reuse training args for shape-related fields
    ns = SimpleNamespace(
        model=ckpt_args.model,
        img_size=ckpt_args.img_size,
        class_num=ckpt_args.class_num,
        attn_dropout=ckpt_args.attn_dropout,
        proj_dropout=ckpt_args.proj_dropout,
        label_drop_prob=ckpt_args.label_drop_prob,
        P_mean=ckpt_args.P_mean,
        P_std=ckpt_args.P_std,
        noise_scale=ckpt_args.noise_scale,
        t_eps=ckpt_args.t_eps,
        ema_decay1=ckpt_args.ema_decay1,
        ema_decay2=ckpt_args.ema_decay2,
        sampling_method=ckpt_args.sampling_method,
        num_sampling_steps=ckpt_args.num_sampling_steps,
        timestep_schedule=ckpt_args.timestep_schedule,
        cfg=ckpt_args.cfg,
        interval_min=ckpt_args.interval_min,
        interval_max=ckpt_args.interval_max,
        ffn_fake_quant=False,
        ffn_use_kl_scales='',
        ffn_int7_weights='',
    )

    denoiser = Denoiser(ns)
    state_key = 'model_ema2' if args.ema_choice == 2 else ('model_ema1' if args.ema_choice == 1 else 'model')
    state = ckpt.get(state_key, ckpt['model'])
    denoiser.load_state_dict(state, strict=True)
    denoiser.to(device)
    denoiser.eval()
    return denoiser


def collect_activation_stats(model: Denoiser, data_loader, device, max_samples: int):
    stats = defaultdict(lambda: defaultdict(list))
    handles = []

    def register_hooks():
        for idx, block in enumerate(model.net.blocks):
            # w12: captures input activations and matmul output
            def _w12_hook(module, inp, out, b_idx=idx):
                x_in = inp[0].detach()
                x_out = out.detach()
                stats[b_idx]['w12_act'].append(x_in.cpu())
                stats[b_idx]['w12_acc'].append(x_out.cpu())

            def _w3_hook(module, inp, out, b_idx=idx):
                hidden_in = inp[0].detach()
                out_acc = out.detach()
                stats[b_idx]['w3_act'].append(hidden_in.cpu())
                stats[b_idx]['w3_acc'].append(out_acc.cpu())

            handles.append(block.mlp.w12.register_forward_hook(_w12_hook))
            handles.append(block.mlp.w3.register_forward_hook(_w3_hook))

    def remove_hooks():
        for h in handles:
            h.remove()

    register_hooks()
    seen = 0

    with torch.no_grad():
        for images, labels in data_loader:
            remaining = max_samples - seen
            if remaining <= 0:
                break
            images = images[:remaining]
            labels = labels[:remaining]
            images = images.to(device, non_blocking=True).float().div_(255).mul_(2.0).sub_(1.0)
            labels = labels.to(device, non_blocking=True)
            _ = model(images, labels)
            seen += images.size(0)

    remove_hooks()
    return stats, seen


def compute_scales_from_stats(stats: dict):
    scales = {}
    for b_idx, tensors in stats.items():
        for name, tensor_list in tensors.items():
            flat = torch.cat([t.flatten() for t in tensor_list]) if tensor_list else torch.tensor([])
            scale = kl_divergence_scale(flat)
            scales[f"blocks.{b_idx}.mlp.{name}"] = float(scale.item())
    return scales


def quantize_ffn_weights(model: Denoiser, clip_percentile: float = 99.8):
    qweights = {}
    for idx, block in enumerate(model.net.blocks):
        qweights[f"blocks.{idx}.mlp.w12.weight"] = quantize_int8_symmetric(block.mlp.w12.weight, clip_percentile)
        qweights[f"blocks.{idx}.mlp.w3.weight"] = quantize_int8_symmetric(block.mlp.w3.weight, clip_percentile)
    return qweights


def get_args():
    parser = argparse.ArgumentParser(description='FFN KL calibration for JiT')
    parser.add_argument('--data_path', type=str, required=True, help='Path to ImageNet root directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint (checkpoint-last.pth)')
    parser.add_argument('--output_scales', type=str, default='ffn_scales_kl.json', help='Where to save KL scales (json)')
    parser.add_argument('--output_qweights', type=str, default='ffn_weights_int8.pt', help='Where to save INT8 weights')
    parser.add_argument('--weight_clip_percentile', type=float, default=99.8, help='Percentile clip for weight quant (0 for MinMax)')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of validation images to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Calibration batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--img_size', type=int, default=256, help='Center crop size')
    parser.add_argument('--ema_choice', type=int, default=2, choices=[0, 1, 2], help='Which weights to load: 0=model, 1=ema1, 2=ema2')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    return parser.parse_args()


def main():
    args = get_args()
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / 'checkpoint-last.pth'
    args.checkpoint = str(ckpt_path)

    device = torch.device(args.device)

    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.CenterCrop(args.img_size),
        transforms.PILToTensor(),
    ])

    dataset = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_denoiser_from_checkpoint(args, device)
    stats, seen = collect_activation_stats(model, loader, device, args.num_samples)
    print(f"Collected activations from {seen} images")

    scales = compute_scales_from_stats(stats)
    qweights = quantize_ffn_weights(model, clip_percentile=args.weight_clip_percentile)

    Path(os.path.dirname(args.output_scales) or '.').mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.output_qweights) or '.').mkdir(parents=True, exist_ok=True)

    with open(args.output_scales, 'w') as f:
        json.dump(scales, f, indent=2)
    torch.save(qweights, args.output_qweights)

    print(f"Saved KL scales to {args.output_scales}")
    print(f"Saved INT8 weights to {args.output_qweights} (clip_pct={args.weight_clip_percentile})")


if __name__ == '__main__':
    main()

# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import os
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm


_DISABLE_TORCH_COMPILE = os.getenv("JIT_DISABLE_TORCH_COMPILE", "0") == "1"
BIT_SERIAL_ADC_BYPASS = os.getenv("BIT_SERIAL_ADC_BYPASS", "0") == "1"


def jit_compile(fn):
    if _DISABLE_TORCH_COMPILE:
        return fn
    return torch.compile(fn)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BitSerialLinearW8A16(nn.Module):
    """
    Bit-serial input CIM-style Linear with INTn weights (default INT8) and virtual INT16 activations.
    Slicing/Reconstruction are forced to float32 to preserve LSB contribution under BF16 autocast.

    NOTE: The historical class name is kept for compatibility, but the default effective activation
    precision now targets W8A12 to model noise budget/overlap redundancy in 2-pass silicon.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, adc_nbit: int = 10, weight_clip_pct: float = 0.0, weight_nbit: int = 8,
                 act_nbit_eff: int = 12, overlap_bits: int | None = None, slice_nbit: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adc_nbit = adc_nbit
        self.weight_clip_pct = weight_clip_pct
        self.weight_nbit = weight_nbit
        self.slice_nbit = slice_nbit
        # Overlap defaults to the gap between the two 8-bit slices and the target effective precision
        default_overlap = max(0, slice_nbit * 2 - act_nbit_eff)
        self.overlap_bits = overlap_bits if overlap_bits is not None else default_overlap
        self.slice_shift = self.slice_nbit - self.overlap_bits
        if self.slice_shift <= 0:
            raise ValueError(f"Invalid overlap_bits={self.overlap_bits}; slice_shift must be positive")
        self.qmax_adc = (1 << (adc_nbit - 1)) - 1
        self.qmin_adc = - (1 << (adc_nbit - 1))
        self.qmax_w = (1 << (weight_nbit - 1)) - 1
        self.qmin_w = - (1 << (weight_nbit - 1))
        self.qmax_act = (1 << (act_nbit_eff - 1)) - 1
        self.qmin_act = - (1 << (act_nbit_eff - 1))
        self.qmax_slice = (1 << (slice_nbit - 1)) - 1
        self.qmin_slice = - (1 << (slice_nbit - 1))

        # Master FP32 weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Optional static buffers (unused in dynamic mode but kept for compatibility)
        self.register_buffer('static_w_int', None)
        self.register_buffer('static_w_scale', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, act_scale: torch.Tensor | None = None, w_int: torch.Tensor | None = None, w_scale: torch.Tensor | None = None):
        # Force float32 to avoid BF16 swallowing LSB contributions
        x = x.float()

        # Optional single-pass (no slicing) control path for debugging
        single_pass = os.getenv("BIT_SERIAL_SINGLE_PASS", "0") == "1"

        # 1) Dynamic weight quant (or use provided)
        if w_int is None or w_scale is None:
            w_in_process = self.weight.float()
            if self.weight_clip_pct > 0.0:
                clip_val = torch.quantile(w_in_process.abs(), self.weight_clip_pct / 100.0)
                w_in_process = torch.clamp(w_in_process, -clip_val, clip_val)
            max_abs_w = torch.max(w_in_process.abs())
            w_scale = torch.clamp(max_abs_w / float(self.qmax_w), min=1e-8)
            w_int = torch.clamp(torch.round(w_in_process / w_scale), self.qmin_w, self.qmax_w)

        w_scale = w_scale.to(dtype=torch.float32, device=x.device)
        w_int = w_int.to(dtype=torch.float32, device=x.device)

        if single_pass:
            # Control path: no slicing, INT8 quant for both act and weight
            act_scale_sp = torch.clamp(torch.max(x.abs()) / 127.0, min=1e-8)
            x_int8 = torch.clamp(torch.round(x / act_scale_sp), -128, 127)
            with torch.cuda.amp.autocast(enabled=False):
                y_raw = F.linear(x_int8, w_int, bias=None)
            y = y_raw * (act_scale_sp * w_scale)
            if self.bias is not None:
                y = y + self.bias.float()
            return y

        # 2) Activation scale: trust provided scale; default to target effective dynamic range when absent
        if act_scale is None:
            max_abs_act = torch.max(x.abs())
            act_scale = torch.clamp(max_abs_act / float(self.qmax_act), min=1e-8)
        else:
            act_scale = act_scale.to(dtype=torch.float32)
        act_scale = act_scale.to(dtype=torch.float32, device=x.device)

        # 3) Quantize activations to INT{act_nbit_eff} in float32 domain
        x_int = torch.clamp(torch.round(x / act_scale), self.qmin_act, self.qmax_act)

        # 4) Bit-slice with overlap (e.g., 2-pass 8-bit slices with 4-bit overlap -> effective 12-bit)
        slice_base = float(2 ** self.slice_shift)
        x_msb = torch.div(x_int, slice_base, rounding_mode='floor').clamp(self.qmin_slice, self.qmax_slice)
        x_lsb = (x_int - x_msb * slice_base).clamp(0, slice_base - 1)

        # 5) MVM in float32 (disable autocast)
        with torch.cuda.amp.autocast(enabled=False):
            y_msb_raw = F.linear(x_msb, w_int, bias=None)
            y_lsb_raw = F.linear(x_lsb, w_int, bias=None)

        # 6) ADC：MSB/LSB 各自独立增益（更贴近双阵列硬件）；可选 bypass
        if BIT_SERIAL_ADC_BYPASS:
            y_msb_adc = y_msb_raw
            y_lsb_adc = y_lsb_raw
        else:
            with torch.no_grad():
                s_adc_msb = y_msb_raw.abs().max() / float(self.qmax_adc) + 1e-8
                s_adc_lsb = y_lsb_raw.abs().max() / float(self.qmax_adc) + 1e-8
            y_msb_adc = torch.clamp(torch.round(y_msb_raw / s_adc_msb), self.qmin_adc, self.qmax_adc)
            y_lsb_adc = torch.clamp(torch.round(y_lsb_raw / s_adc_lsb), self.qmin_adc, self.qmax_adc)
            y_msb_adc = y_msb_adc * s_adc_msb
            y_lsb_adc = y_lsb_adc * s_adc_lsb

        # 7) Reconstruct with configured overlap (e.g., 4-bit overlap -> shift 4), then apply real scale
        slice_scale = float(2 ** self.slice_shift)
        global_scale = act_scale * w_scale * slice_scale
        y = y_msb_adc.to(torch.float32) * global_scale + y_lsb_adc.to(torch.float32) * (global_scale / slice_scale)

        if self.bias is not None:
            y = y + self.bias.float()
        return y


class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype).cuda()

    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True,
        bitserial: bool = True,
        static_scales: dict | None = None,
        static_qweights: dict | None = None,
        weight_clip_pct: float = 0.0,
        weight_nbit: int = 8,
        act_nbit_eff: int = 12,
        overlap_bits: int | None = None,
        adc_nbit: int = 10,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        # 默认启用位串行，走等效 W8A12（可调激活精度）的 2-pass 位切分；如需禁用则退回常规 Linear。
        self.use_bitserial = bitserial
        self.weight_nbit = weight_nbit
        if self.use_bitserial:
            self.w12 = BitSerialLinearW8A16(dim, 2 * hidden_dim, bias=bias, adc_nbit=adc_nbit, weight_clip_pct=weight_clip_pct, weight_nbit=weight_nbit,
                                            act_nbit_eff=act_nbit_eff, overlap_bits=overlap_bits)
            self.w3 = BitSerialLinearW8A16(hidden_dim, dim, bias=bias, adc_nbit=adc_nbit, weight_clip_pct=weight_clip_pct, weight_nbit=weight_nbit,
                                           act_nbit_eff=act_nbit_eff, overlap_bits=overlap_bits)
        else:
            self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
            self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)
        self.static_scales = static_scales or {}
        self.static_qweights = static_qweights or {}
        self.weight_clip_pct = weight_clip_pct

    def forward(self, x):
        if self.use_bitserial:
            act_scale_w12 = self.static_scales.get('w12_act')
            if act_scale_w12 is not None:
                act_scale_w12 = act_scale_w12.to(device=x.device, dtype=torch.float32) if torch.is_tensor(act_scale_w12) else torch.tensor(act_scale_w12, device=x.device, dtype=torch.float32)

            w12_int = None
            w12_scale = None
            if 'w12.weight' in self.static_qweights:
                w12_q = self.static_qweights['w12.weight']
                w12_q = w12_q.to(device=x.device, dtype=torch.float32)
                self.static_qweights['w12.weight'] = w12_q
                max_abs_w = torch.max(w12_q.abs())
                w12_scale = torch.clamp(max_abs_w / float(self.w12.qmax_w), min=1e-8)
                w12_int = torch.clamp(torch.round(w12_q / w12_scale), self.w12.qmin_w, self.w12.qmax_w)

            x12 = self.w12(x, act_scale=act_scale_w12, w_int=w12_int, w_scale=w12_scale)
        else:
            x12 = self.w12(x)

        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2

        if self.use_bitserial:
            act_scale_w3 = self.static_scales.get('w3_act')
            if act_scale_w3 is not None:
                act_scale_w3 = act_scale_w3.to(device=hidden.device, dtype=torch.float32) if torch.is_tensor(act_scale_w3) else torch.tensor(act_scale_w3, device=hidden.device, dtype=torch.float32)

            w3_int = None
            w3_scale = None
            if 'w3.weight' in self.static_qweights:
                w3_q = self.static_qweights['w3.weight']
                w3_q = w3_q.to(device=hidden.device, dtype=torch.float32)
                self.static_qweights['w3.weight'] = w3_q
                max_abs_w = torch.max(w3_q.abs())
                w3_scale = torch.clamp(max_abs_w / float(self.w3.qmax_w), min=1e-8)
                w3_int = torch.clamp(torch.round(w3_q / w3_scale), self.w3.qmin_w, self.w3.qmax_w)

            out = self.w3(hidden, act_scale=act_scale_w3, w_int=w3_int, w_scale=w3_scale)
        else:
            out = self.w3(hidden)

        return out


class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    @jit_compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, bitserial_ffn: bool = True,
                 ffn_static_scales: dict | None = None, ffn_static_qweights: dict | None = None, ffn_weight_clip_pct: float = 0.0,
                 ffn_weight_nbit: int = 8, ffn_act_nbit_eff: int = 12, ffn_overlap_bits: int | None = None, ffn_adc_nbit: int = 10):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop, bitserial=bitserial_ffn,
                             static_scales=ffn_static_scales, static_qweights=ffn_static_qweights,
                             weight_clip_pct=ffn_weight_clip_pct, weight_nbit=ffn_weight_nbit,
                             act_nbit_eff=ffn_act_nbit_eff, overlap_bits=ffn_overlap_bits, adc_nbit=ffn_adc_nbit)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    @jit_compile
    def forward(self, x,  c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class JiT(nn.Module):
    """
    Just image Transformer.
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        ffn_bitserial: bool = True,
        ffn_scales_path: str = '',
        ffn_int7_weights_path: str = '',
        ffn_weight_clip_pct: float = 0.0,
        ffn_weight_nbit: int = 8,
        ffn_act_nbit_eff: int = 12,
        ffn_overlap_bits: int | None = None,
        ffn_adc_nbit: int = 10,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes
        self.ffn_weight_clip_pct = ffn_weight_clip_pct
        self.ffn_act_nbit_eff = ffn_act_nbit_eff
        self.ffn_overlap_bits = ffn_overlap_bits
        self.ffn_adc_nbit = ffn_adc_nbit

        # time and class embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # linear embed
        self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)

        # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # in-context cls token
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        # rope
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len
        )

        # transformer
        # Optional static quant assets
        ffn_scales_all = self._load_ffn_static(ffn_scales_path)
        ffn_qweights_all = self._load_ffn_qweights(ffn_int7_weights_path)

        self.blocks = nn.ModuleList([])
        for i in range(depth):
            scales_i = self._slice_ffn_static(ffn_scales_all, i)
            qweights_i = self._slice_ffn_static(ffn_qweights_all, i)
            block = JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                             attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                             proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                             bitserial_ffn=ffn_bitserial,
                             ffn_static_scales=scales_i if scales_i else None,
                             ffn_static_qweights=qweights_i if qweights_i else None,
                             ffn_weight_clip_pct=ffn_weight_clip_pct,
                             ffn_weight_nbit=ffn_weight_nbit,
                             ffn_act_nbit_eff=ffn_act_nbit_eff,
                             ffn_overlap_bits=ffn_overlap_bits,
                             ffn_adc_nbit=ffn_adc_nbit)
            self.blocks.append(block)

        # linear predict
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _load_ffn_static(self, path: str):
        if not path:
            return {}
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"FFN static scale file not found: {path}")
        ext = path_obj.suffix.lower()
        if ext == '.json':
            with open(path_obj, 'r') as f:
                data = json.load(f)
        elif ext == '.npz':
            data = dict(np.load(path_obj, allow_pickle=False))
        else:
            raise ValueError(f"Unsupported ffn_scales_path extension: {ext}")
        scales = {}
        for k, v in data.items():
            if isinstance(v, (list, tuple)):
                v = np.array(v)
            v_t = torch.tensor(v)
            scales[k] = v_t
        return scales

    def _load_ffn_qweights(self, path: str):
        if not path:
            return {}
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"FFN INT7 weight file not found: {path}")
        state = torch.load(path_obj, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        return state

    @staticmethod
    def _slice_ffn_static(mapping: dict, block_idx: int):
        if not mapping:
            return {}
        prefix = f"blocks.{block_idx}.mlp."
        alt_prefix = f"module.{prefix}"
        alt_prefix2 = f"net.{prefix}"
        sliced = {}
        for k, v in mapping.items():
            key = k
            if key.startswith(alt_prefix):
                key = key[len(alt_prefix):]
            elif key.startswith(alt_prefix2):
                key = key[len(alt_prefix2):]
            elif key.startswith(prefix):
                key = key[len(prefix):]
            else:
                continue
            sliced[key] = v
        return sliced

    def unpatchify(self, x, p):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        x: (N, C, H, W)
        t: (N,)
        y: (N,)
        """
        # class and time embeddings
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        # forward JiT
        x = self.x_embedder(x)
        x += self.pos_embed

        for i, block in enumerate(self.blocks):
            # in-context
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens += self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
            x = block(x, c, self.feat_rope if i < self.in_context_start else self.feat_rope_incontext)

        x = x[:, self.in_context_len:]

        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)

        return output


def JiT_B_16(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=16, **kwargs)

def JiT_B_32(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=32, **kwargs)

def JiT_L_16(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=16, **kwargs)

def JiT_L_32(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=32, **kwargs)

def JiT_H_16(**kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=16, **kwargs)

def JiT_H_32(**kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=32, **kwargs)


JiT_models = {
    'JiT-B/16': JiT_B_16,
    'JiT-B/32': JiT_B_32,
    'JiT-L/16': JiT_L_16,
    'JiT-L/32': JiT_L_32,
    'JiT-H/16': JiT_H_16,
    'JiT-H/32': JiT_H_32,
}

# 工程功能梳理

本仓库在 Kaiming He 的 JiT 图像生成模型基础上，加入了面向类模拟内存计算(AIMC)硬件的量化与位串行线性层模拟。下面从整体流程、核心模型与量化/ADC 实现三个角度进行说明。

## 目录与主流程
- `main_jit.py`：命令行参数、数据集加载与分布式初始化，调用 `engine_jit.py` 完成训练/评估，并支持在线采样/生成。关键参数包含 CFG、时间步调度、EMA、FFN 量化开关与外部量化资产路径等。【F:main_jit.py†L23-L149】
- `denoiser.py`：包装 `JiT` 主干，负责生成噪声/时间步、CFG 调度以及 Heun/Euler 采样器；训练时输出扩散损失，推理时用于图像生成。【F:denoiser.py†L6-L181】
- `calibrate_kl.py`：收集前向激活统计并用 KL 散度搜索量化 scale，保存为静态校准文件；同时提供权重量化与从 checkpoint 构建推理用 `Denoiser` 的工具函数。【F:calibrate_kl.py†L19-L199】
- `model_jit.py`：实现 JiT 模型、AIMC 友好的位串行线性层，以及所有 Transformer 组件（补丁嵌入、RMSNorm、RoPE 等）。【F:model_jit.py†L1-L520】

## 核心组件与量化/ADC 实现
### BitSerialLinearW8A10（Nor-Flash 6+4 位切分）
- 目标：用 INT8 权重、INT10 激活的 **6-bit MSB + 4-bit LSB** 2-pass 流程模拟 Nor-Flash CIM，默认 10bit（可配置 10~12bit）ADC，并支持 MSB 多次采样降噪与 LSB DAC 4× 增益。【F:model_jit.py†L32-L190】
- 主要步骤：
  1. **动态权重量化**：量化到映射到 [-128,127]；【F:model_jit.py†L120-L129】
  2. **激活 INT10 量化与 6/4 分拆**：动态 99.5% 分位（默认值）对称截断后量化到 [-512,511]，再用算术右移取 6-bit 有符号 MSB、按位与提取 4-bit 无符号 LSB，并在 DAC 侧将 LSB 左移 2bit（4× 增益）。【F:model_jit.py†L131-L154】
  3. **模拟阵列 MVM**：MSB 通道可重复采样 K 次（默认 2、上限 4）并平均以降噪，LSB 通道使用增益后的输入；均用 FP32 计算以保持精度。【F:model_jit.py†L156-L165】
  4. **ADC 量化**：对 MSB/LSB raw 结果分别量化到 N-bit ADC（默认 10bit，支持静态 scale 或 bypass），静态单值 scale 视作重建域步长并自动折算到各通道累加器域。【F:model_jit.py†L167-L187】
  5. **数字域重构**：按公式 \(\hat{y} = 16 \cdot \overline{y_H} + y'_L/4\) 组合两通道，再乘以激活/权重全局 scale，最后加偏置。【F:model_jit.py†L189-L192】

### FFN 位串行与静态量化路径
- `SwiGLUFFN` 在 `bitserial=True` 时用两层 `BitSerialLinearW8A10`（固定 INT10 6/4 分拆，支持 MSB 采样次数、LSB 增益与 ADC 位宽配置）模拟前馈；否则回退到普通 `nn.Linear`。【F:model_jit.py†L315-L402】
- 支持两种模式：
- **动态位串行**：默认动态权重量化、99.5% 分位激活截断、MSB 多次采样与 LSB 增益，配合 ADC 饱和；可通过 `BIT_SERIAL_SINGLE_PASS` 走不切片的调试路径。【F:model_jit.py†L117-L143】【F:model_jit.py†L131-L190】
- **静态量化**：若传入 KL 校准的 `static_scales` 与预量化权重，则按固定 scale 执行位串行重建（激活 scale 以 FP32 传入，权重量化可由预量化权重反推 scale 与整数权重）。其中 `*_acc` 的单值 scale 默认视作“重建后输出”的量化步长，内部会自动折算到 MSB/LSB raw 累加域再送入 ADC。【F:model_jit.py†L320-L402】【F:model_jit.py†L146-L184】

#### 动态 vs. 静态量化与 KL 校准
- **动态量化**（运行时自适应）
  - 权重与激活 scale 都在 `forward` 内按当前 batch 最大绝对值或百分位裁剪即时计算，KL 不介入；ADC 也用运行时峰值自适应。【F:model_jit.py†L117-L183】
  - 适合快速实验或硬件不要求固定 scale 的场景，推理时无需外部校准文件。
- **静态量化**（离线定标）
  - 使用 `calibrate_kl.py` 在代表性数据上收集 Hook 的激活/累加器分布，`kl_divergence_scale` 计算最佳截断阈值 -> 生成 `w12_act/w12_acc/w3_act/w3_acc` 等固定 scale。【F:calibrate_kl.py†L19-L199】
  - FFN 前馈时直接加载这些 scale，并可配合 `quantize_ffn_weights` 导出的 INT8 权重，实现全链路固定量化；ADC 采用预估的 `*_acc` scale 而非运行时峰值，保持等效 INT12 量化与噪声预算假设一致。【F:model_jit.py†L320-L402】【F:model_jit.py†L150-L183】
  - 因为 scale 是离线确定的，静态路径与动态路径的 KL 校准并不共享：只有静态模式需要运行 KL 校准并加载结果，动态模式完全依赖运行时数值范围。

### KL 校准流程
- `collect_activation_stats` 在每个 FFN 线性层注册 hook，收集输入与累加器输出，再拼接后用于求 KL 最优 scale；保存格式与 `blocks.{idx}.mlp.{name}` 对应模型层。【F:calibrate_kl.py†L147-L199】
- `kl_divergence_scale` 遍历阈值，寻找最小 KL 的直方图截断点并转换为对称 INT8 scale，服务于静态激活量化。【F:calibrate_kl.py†L19-L69】
- 权重可用 `quantize_int8_symmetric` 做百分位裁剪+对称 INT8 量化，生成静态权重量化文件。【F:calibrate_kl.py†L72-L82】

### 采样与CFG
- `Denoiser.generate` 构造线性或 logit-normal 时间步，按 Heun/Euler ODE 积分；`_forward_sample` 同时走有/无条件分支并在区间内应用 CFG 缩放，便于与硬件量化路径解耦。【F:denoiser.py†L100-L181】

## 量化/硬件相关小贴士
- 环境变量 `BIT_SERIAL_ADC_BYPASS` 可跳过 ADC 量化，`BIT_SERIAL_SINGLE_PASS` 可禁用位切分；`JIT_DISABLE_TORCH_COMPILE` 可关闭 `torch.compile` 以方便调试。【F:model_jit.py†L18-L25】【F:model_jit.py†L117-L119】【F:model_jit.py†L164-L176】
- `main_jit.py` 暴露 `--[no-]ffn_bitserial/--ffn_use_kl_scales/--ffn_int7_weights/--ffn_weight_clip_pct/--ffn_act_nbit/--ffn_msb_samples/--ffn_lsb_gain_shift/--ffn_adc_nbit`，默认走固定 INT10（6/4 分拆）位串行路径，可在推理时灵活切换动态/静态量化、MSB 采样与 LSB 增益。【F:main_jit.py†L109-L152】
- 位串行 MSB 通道在 ADC 量化前默认注入 5 LSB（单次采样）的高斯热噪声，并按 `msb_samples` 开根号衰减，模拟多次采样平均后的等效噪声。这里计算过程中的热噪声全部等效到ADC采样端（对于10 bit的ADC来说，动态范围相当于1024 LSB，热噪声标准差则为5LSB）【F:model_jit.py†L269-L299】

## ImageNet FID-50k 评估流程（KL 静态校准无噪声，推理加噪 2.0）
1. **KL 静态校准（无 MSB 噪声）**：先为 FFN 线性层收集激活/累加器直方图并搜索 KL 最优 scale，注意将 `ffn_msb_noise_sigma_lsb` 设为 `0.0`，避免把噪声写入静态标定文件。
   ```bash
   python calibrate_kl.py \
     --data_path /path/to/imagenet/val \
     --output_scales /path/to/ffn_scales_imagenet.npz \
     --ffn_bitserial \
     --ffn_msb_noise_sigma_lsb 0.0
   ```
   如需同时导出量化权重，可追加对应的权重量化输出参数。

2. **FID-50k/IS 推理评估（MSB 噪声 2.0）**：加载上一步生成的 KL 静态 scale（及可选 INT7/INT8 权重），在推理时开启 `ffn_msb_noise_sigma_lsb=2.0`，其余位宽/采样与硬件匹配。
   ```bash
   python main_jit.py \
     --data_path /path/to/imagenet/val \
     --fid50k \
     --ffn_bitserial \
     --ffn_use_kl_scales /path/to/ffn_scales_imagenet.npz \
     --ffn_msb_noise_sigma_lsb 5.0 \
     --ffn_adc_nbit 10 \
     --ffn_act_nbit 10 \
     --ffn_msb_samples 2 \
     --ffn_lsb_gain_shift 2
   ```
   - 确保评估时的噪声只作用在推理前向，静态校准文件保持无噪声版本。
   - 若同时加载静态量化权重，请带上对应的权重路径并与 scale 一起使用。

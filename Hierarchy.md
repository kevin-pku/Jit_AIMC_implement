## 目录与主流程
- `main_jit.py`：命令行参数、数据集加载与分布式初始化，调用 `engine_jit.py` 完成训练/评估，并支持在线采样/生成。关键参数包含 CFG、时间步调度、EMA、FFN 量化开关与外部量化资产路径等。【F:main_jit.py†L23-L149】
- `denoiser.py`：包装 `JiT` 主干，负责生成噪声/时间步、CFG 调度以及 Heun/Euler 采样器；训练时输出扩散损失，推理时用于图像生成。【F:denoiser.py†L6-L181】
- `calibrate_kl.py`：收集前向激活统计并用 KL 散度搜索量化 scale，保存为静态校准文件；同时提供权重量化与从 checkpoint 构建推理用 `Denoiser` 的工具函数。【F:calibrate_kl.py†L19-L199】
- `model_jit.py`：实现 JiT 模型、AIMC 友好的位串行线性层，以及所有 Transformer 组件（补丁嵌入、RMSNorm、RoPE 等）。【F:model_jit.py†L1-L520】

## 核心组件与量化/ADC 实现
### BitSerialLinearW8A12（带重叠的位串行线性层）
- 目标：用 INT8 权重、带 4bit 重叠的双 8bit slice 实现 **等效 W8A12**，以建模 2-Pass CIM 在噪声预算/ENOB 限制下的有效 12bit 精度，同时仍考虑默认 10bit（可配置 10~12bit）ADC 饱和与可选 bypass。【F:model_jit.py†L81-L186】
- 主要步骤：
  1. **动态权重量化**：按最大绝对值求 scale，映射到 [-128,127]；可选百分位裁剪以抑制异常值。【F:model_jit.py†L120-L129】
  2. **激活 INT12 量化与重叠位切分**：默认将输入量化到 [-2048,2047]，采用 8bit slice + 4bit overlap（可通过 `--ffn_act_nbit/--ffn_overlap_bits` 调整）生成 MSB/LSB，模拟为抵抗底噪而牺牲的冗余位宽。【F:model_jit.py†L135-L158】
  3. **模拟阵列 MVM**：关闭 autocast 用 FP32 计算 MSB/LSB 线性输出，确保低位精度。【F:model_jit.py†L159-L163】
  4. **ADC 量化**：分别对 MSB/LSB 结果按自身峰值映射到 N-bit ADC（默认 10bit，支持环境变量 bypass），再反量化回模拟电平。【F:model_jit.py†L164-L176】
  5. **重建与缩放**：使用重叠切分对应的步长重组 MSB/LSB，再乘以激活/权重全局 scale；最后加上偏置。【F:model_jit.py†L177-L186】

### FFN 位串行与静态量化路径
- `SwiGLUFFN` 在 `bitserial=True` 时用两层 `BitSerialLinearW8A16`（默认等效 W8A12，可配置有效位宽、重叠与 ADC 位宽）模拟前馈；否则回退到普通 `nn.Linear`。【F:model_jit.py†L320-L402】
- 支持两种模式：
  - **动态位串行**：默认动态权重量化与激活动态 scale，配合 ADC 饱和；可通过 `BIT_SERIAL_SINGLE_PASS` 走不切片的调试路径。【F:model_jit.py†L117-L143】【F:model_jit.py†L144-L183】
  - **静态量化**：若传入 KL 校准的 `static_scales` 与预量化权重，则按固定 scale 执行位串行重建（激活 scale 以 FP32 传入，权重量化可由预量化权重反推 scale 与整数权重）。【F:model_jit.py†L320-L402】

#### 动态 vs. 静态量化与 KL 校准
- **动态量化**（运行时自适应）
  - 权重与激活 scale 都在 `forward` 内按当前 batch 最大绝对值或百分位裁剪即时计算，KL 不介入；ADC 也用运行时峰值自适应。【F:model_jit.py†L117-L183】
  - 适合快速实验或硬件不要求固定 scale 的场景，推理时无需外部校准文件。
- **静态量化**（离线定标）
  - 使用 `calibrate_kl.py` 在代表性数据上收集 Hook 的激活/累加器分布，`kl_divergence_scale` 计算最佳截断阈值 -> 生成 `w12_act/w12_acc/w3_act/w3_acc` 等固定 scale。【F:calibrate_kl.py†L19-L199】
  - FFN 前馈时直接加载这些 scale，并可配合 `quantize_ffn_weights` 导出的 INT8 权重，实现全链路固定量化；ADC 也按静态 `*_acc` scale 仿真。【F:model_jit.py†L336-L402】
  - 因为 scale 是离线确定的，静态路径与动态路径的 KL 校准并不共享：只有静态模式需要运行 KL 校准并加载结果，动态模式完全依赖运行时数值范围。

### KL 校准流程
- `collect_activation_stats` 在每个 FFN 线性层注册 hook，收集输入与累加器输出，再拼接后用于求 KL 最优 scale；保存格式与 `blocks.{idx}.mlp.{name}` 对应模型层。【F:calibrate_kl.py†L147-L199】
- `kl_divergence_scale` 遍历阈值，寻找最小 KL 的直方图截断点并转换为对称 INT8 scale，服务于静态激活量化。【F:calibrate_kl.py†L19-L69】
- 权重可用 `quantize_int8_symmetric` 做百分位裁剪+对称 INT8 量化，生成静态权重量化文件。【F:calibrate_kl.py†L72-L82】

### 采样与CFG
- `Denoiser.generate` 构造线性或 logit-normal 时间步，按 Heun/Euler ODE 积分；`_forward_sample` 同时走有/无条件分支并在区间内应用 CFG 缩放，便于与硬件量化路径解耦。【F:denoiser.py†L100-L181】

## 量化/硬件相关小贴士
- 环境变量 `BIT_SERIAL_ADC_BYPASS` 可跳过 ADC 量化，`BIT_SERIAL_SINGLE_PASS` 可禁用位切分；`JIT_DISABLE_TORCH_COMPILE` 可关闭 `torch.compile` 以方便调试。【F:model_jit.py†L18-L25】【F:model_jit.py†L117-L119】【F:model_jit.py†L164-L176】
- `main_jit.py` 暴露 `--[no-]ffn_bitserial/--ffn_use_kl_scales/--ffn_int7_weights/--ffn_weight_clip_pct/--ffn_act_nbit/--ffn_overlap_bits/--ffn_adc_nbit`，默认走等效 W8A12 位串行路径，可在推理时灵活切换动态/静态量化、有效位宽与权重裁剪策略。【F:main_jit.py†L109-L151】

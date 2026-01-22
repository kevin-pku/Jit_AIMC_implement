# 工程功能梳理

本仓库在 Kaiming He 的 JiT 图像生成模型基础上，加入了面向类模拟内存计算(AIMC)硬件的量化与位串行线性层模拟。下面从整体流程、核心模型与量化/ADC 实现三个角度进行说明。

## 目录与主流程
- `main_jit.py`：命令行参数、数据集加载与分布式初始化，调用 `engine_jit.py` 完成训练/评估，并支持在线采样/生成。关键参数包含 CFG、时间步调度、EMA、FFN 量化开关与外部量化资产路径等。【F:main_jit.py†L23-L149】
- `denoiser.py`：包装 `JiT` 主干，负责生成噪声/时间步、CFG 调度以及 Heun/Euler 采样器；训练时输出扩散损失，推理时用于图像生成。【F:denoiser.py†L6-L181】
- `calibrate_kl.py`：收集前向激活统计并用 KL 散度搜索量化 scale，保存为静态校准文件；同时提供权重量化与从 checkpoint 构建推理用 `Denoiser` 的工具函数。【F:calibrate_kl.py†L19-L199】
- `model_jit.py`：实现 JiT 模型、AIMC 友好的位串行线性层，以及所有 Transformer 组件（补丁嵌入、RMSNorm、RoPE 等）。【F:model_jit.py†L1-L520】


## 核心组件与量化/ADC 实现

### HybridLinearW8A11（数模混合稀疏残差架构）

* **设计目标**：在受限的激活分布下实现 **等效 INT8 系统精度**（INT8-equivalent system precision）。架构采用 **2-bit 数字 MSB (稀疏)** + **9-bit 模拟 LSB (存算)** 的混合路径，利用激活值的本征长尾分布特性，将模拟噪声限制在低权重分量，并通过无噪声的数字域补偿高权重残差，从而在物理层面规避传统位串行方案（如 6+4）中的噪声放大问题。

* **统计学依据 (Statistical Convergence)**：
  本架构基于大规模激活值统计设计。通过采集 **208 个样本**（Spread 标签设定），对 RMSNorm 后约 **19.5 亿 (1.95B)** 个 INT 量化值进行的覆盖率分析显示：

  * **[-256, 255] 区间覆盖率**：**92.04%**（适配 9-bit LSB）。
  * **[-128, 127] 区间覆盖率**：**77.29%**（不足以支撑 8-bit LSB）。
  * **结论**：MSB=0 的条件在绝大多数激活值中成立。这种稀疏性源于数据的**本征分布**而非激进截断，验证了数字残差补偿路径的工程有效性。

* **主要步骤**：

  1. **动态权重量化**：
     权重映射到 $[-128, 127]$ (INT8 标准)。【F:model_jit.py】

  2. **激活 INT11 量化与中心化残差切分 (Center-Out Slicing)**：

     * **量化**：基于动态统计（如 99.8% 分位）将激活值对称量化至 $[-1024, 1023]$ (INT11)。
     * **残差切分策略**：放弃传统的按位切分，采用基于数值区间的残差分解以最大化稀疏度。

       * **模拟 LSB**：Clamp 至 $[-256, 255]$ (9-bit 有符号数)。该区间覆盖了 92.04% 的数据，使模拟阵列工作在最佳线性区。
       * **数字 MSB**：计算残差 $MSB = \text{round_down}((x - LSB) / 256)$。
       * **编码**：MSB 表示为 **2-bit 有符号整数** ${-3, ..., 3}$。在 >92% 的情况下 MSB 恒为 0，数字 MAC 处于空闲（Skip）状态，实现极低功耗。

  3. **混合路径 MVM (Mixed-Signal MVM)**：

     * **路径 A (模拟 LSB)**：输入范围 $[-256, 255]$。利用 Nor-Flash 阵列计算。由于 9-bit 输入直接匹配 10-bit ADC 的动态范围，**取消 4× 增益移位**，实现 1:1 映射以避免底噪放大。
     * **路径 B (数字 MSB)**：仅对非零 MSB（约 8% 的离群值）执行高精度数字乘加 (INT8 MAC)。此路径完全无噪声。

  4. **ADC 量化**：
     对 LSB 通道的模拟累加结果进行量化。推荐配置 **10-bit ADC**。保留 1-bit 的动态余量（9-bit 信号 vs 10-bit ADC）以抵消积分非线性 (INL) 和微分非线性 (DNL) 误差。【F:model_jit.py】

  5. **无噪声放大重构**：
     最终输出重构公式：
     $$\hat{y} = (256 \cdot y_{\text{MSB_digital}}) + y_{\text{LSB_analog}}$$
     **关键改进**：放大系数 ($256$) 仅作用于无噪声的数字结果。模拟分量的增益为 1，确保系统信噪比 (SNR) 仅由 LSB 路径的本征性能决定，未引入结构性噪声放大。

### FFN 混合位串行与静态量化路径

* `SwiGLUFFN` 在 `hybrid_mode=True` 时启用 `HybridLinearW8A11` 混合架构。
* **支持两种模式**：

  * **动态混合模式 (Dynamic Hybrid)**：
    运行时实时统计激活分布。仅当激活值超出 $\pm 256$ 线性区时动态触发数字 MSB 路径。该模式能自适应不同层级的分布偏移 (Distribution Shift)，同时最小化数字能耗。
  * **静态校准模式 (Static Calibrated)**：
    利用 KL 散度校准预计算 `static_scales`，固定量化步长和 MSB/LSB 的稀疏模式。适用于推理部署，以减少运行时的动态分支判断开销。【F:model_jit.py】


-   **支持两种模式**：
    -   **动态混合模式 (Dynamic Hybrid)**：
        运行时实时统计激活分布。仅当激活值超出 $\pm 256$ 线性区时动态触发数字 MSB 路径。该模式能自适应不同层级的分布偏移 (Distribution Shift)，同时最小化数字能耗。
    -   **静态校准模式 (Static Calibrated)**：
        利用 KL 散度校准预计算 `static_scales`，固定量化步长和 MSB/LSB 的稀疏模式。适用于推理部署，以减少运行时的动态分支判断开销。【F:model_jit.py】
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

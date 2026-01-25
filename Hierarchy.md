# 工程功能梳理

本仓库对于 Kaiming He 的 JiT 图像生成模型中的线性层运算，基于**模拟存算一体 (AIMC)** 硬件实现的矩阵向量乘法（MVM）进行模拟。以下从整体流程、核心模型架构与量化/ADC 实现三个维度进行说明。

## 1. 目录结构与主流程

- **`main_jit.py`**
  负责命令行参数解析、数据集加载与分布式初始化。调用 `engine_jit.py` 完成训练或评估，并支持在线采样/生成。
  - **关键参数**：CFG（分类器指导）、时间步调度、EMA、FFN 量化开关及外部量化资产路径等。

- **`denoiser.py`**
  封装 `JiT` 主干网络，负责生成噪声/时间步、CFG 调度以及 Heun/Euler 采样器。
  - **功能**：训练时输出扩散损失，推理时执行图像生成。

- **`calibrate_kl.py`**
  量化校准工具。负责收集前向激活统计数据，利用 KL 散度搜索最优量化 Scale 并保存为静态校准文件；同时提供权重量化与从 Checkpoint 构建推理用 `Denoiser` 的工具函数。

- **`model_jit.py`**
  核心模型实现。包含 JiT 模型架构、AIMC 友好的位串行线性层，以及所有 Transformer 组件（Patch Embedding、RMSNorm、RoPE 等）。

## 2. 核心组件与量化/ADC 实现

### Hybrid_W8A10（模拟-稀疏数字混合架构）

#### 设计目标
在模拟存算一体信噪比受限下，实现 高能效的 **INT8 等效系统精度 (INT8-equivalent precision)**。该架构中权重为INT8动态范围，对动态范围要求更高的激活向量Act，则为INT10动态范围。为了提高在模拟计算中act的实际精度和信噪比，采用 **2-bit 稀疏数字 MSB** + **8-bit 模拟 LSB** 的混合路径，利用激活值的本征长尾分布特性，最大化能效比。

#### 动态机制：稀疏旁路与非零编码 (Zero-Skipping & Non-Zero Encoding)
本架构引入了 **稀疏控制逻辑**，对于每一组输入act（已经先经过99.97pct的动态clipping），INT10量化并进行数值范围判断（具体以补码为例，只要10位二进制数的前3位是全0 (000) 或全1 (111)，数据必然落在[-128, 127]，这是充要条件）：

1.  **Mode 0: 稀疏旁路 (Bypass Mode)**
    *   **触发条件**：当激活值落在 $\rightarrow [-128, 127]$ 区间（占数据总量 >92%）。
    *   **行为**：稀疏控制器判定 MSB 为 0，**直接关断数字 MSB 路径**，仅由模拟 LSB 核心输出结果。此时 MSB 处于关断状态，不消耗数字功耗。

2.  **Mode 1: 非零编码 (Non-Zero Encoding)**
    *   **触发条件**：当激活值属于长尾离群值
    *   **行为**：激活数字路径。由于“0”状态已由 Mode 0 处理，**2-bit MSB 负载无需包含“0”值**。我们将 2-bit 编码重定义为 4 个非零状态 **$\{-2, -1, 1, 2\}$**。
    *   在数字域完成简单移位和运算（由于MSB只有两位，这里的运算其实是对于权重的简单移位加法），等LSB数据被ADC采样后，MSB数据与LSB数据在数字域做移位和
    *   特别注意：在 Mode 1 下，模拟阵列接收的 LSB 并不是原始数据（如果是补码）的低8位（Bits[7:0]），需要做一定的映射（这在具体芯片中通常由 DAC 码表或简单的数字加减处理，不影响架构，但一定要留意）

#### 动态范围重构 (Scale-256 Strategy)
配合 **Scale=256** 的步长与 $\rightarrow [-128, 127]$ 的 LSB 补码能力，实现INT10全域连续覆盖，又保留了更高的精度：

*   **中心区 (MSB OFF)**：直接输出 LSB $\rightarrow [-128, 127]$。
*   **正向扩展 (MSB=1，2)**：
    *   MSB=1: $256 + [-128, 127] \rightarrow [128, 383]$
    *   MSB=2: $512 + [-128, 127] \rightarrow [384, 639]$ (**覆盖 INT10 全范围**)
*   **负向保护 (MSB=-1，-2)**： 同理，向下覆盖，满足INT10动态范围要求。

#### 混合路径处理流程
1.  **动态权重量化**：权重映射到 $[-128, 127]$ (INT8 标准)。
2.  **模拟路径 (LSB)**：
    *   输入范围 $[-128, 127]$，利用 Nor-Flash 电流镜阵列计算，由8-bit DAC提供模拟输入，由10-bit ADC完成模拟采样（实际采样的有效精度为9bit），随后在数字域进行移位处理。
3.  **数字路径 (MSB)**：
    *   仅对 Mode 1 的非零 MSB 执行高精度数字乘加。
4.  **ADC 量化**：
    *   对 LSB 通道累加结果进行 10-bit 量化，保留 1-bit 动态余量以抵消积分非线性 (INL) 误差。
5.  **无噪声放大重构**：
    y = 256 × MSB_digital + LSB_analog

`SwiGLUFFN` 在开启 `hybrid_mode=True` 时启用上述混合架构。

### 量化clipping
对于激活值和权重值，有两种量化模式
1.  **动态clipping模式**：
    默认模式，运行时进行实时统计分布，根据特定百分比pct进行饱和截断
2.  **静态校准模式 (Static Calibrated)**：
    利用 KL 散度预计算 `static_scales`，固定量化步长和稀疏模式。适用于未来推理部署。

### 采样与 CFG
`Denoiser.generate` 构造线性或 Logit-Normal 时间步，按 Heun/Euler ODE 积分。`_forward_sample` 同时执行有/无条件分支，并在区间内应用 CFG 缩放，逻辑与硬件量化路径解耦。

## 3. 硬件仿真配置指南

-   **环境变量**：
    -   `BIT_SERIAL_ADC_BYPASS`：跳过 ADC 量化。
    -   `BIT_SERIAL_SINGLE_PASS`：禁用位切分（调试用）。
    -   `JIT_DISABLE_TORCH_COMPILE`：关闭 `torch.compile`。

-   **关键参数 (`main_jit.py`)**：
    混合模式下遵循上述 2+8 架构。
    可通过 `--ffn_msb_samples` 和 `--ffn_lsb_gain_shift` 调整噪声与增益。

-   **噪声模拟**：
    ADC 采样端默认注入高斯热噪声（标准差 5 LSB @ 10-bit ADC），并按 `msb_samples` 开根号衰减，模拟多次采样平均效果。

## FID-50k 评估流程（注入噪声）
在推理时开启噪声模拟（此处示例为 10 LSB），其余位宽/采样配置需与硬件设计匹配。

python main_jit.py \
  --data_path /path/to/imagenet/val \
  --fid50k \
  --ffn_bitserial \
  --ffn_use_kl_scales /path/to/ffn_scales_imagenet.npz \ （KL散度量化可选）
  --ffn_msb_noise_sigma_lsb 10 \
  --ffn_adc_nbit 10 \
  --ffn_act_nbit 10 \
  --ffn_msb_samples 2 \

## 附录：KL 静态校准（待探索）
```bash
python calibrate_kl.py \
  --data_path /path/to/imagenet/val \
  --output_scales /path/to/ffn_scales_imagenet.npz \
  --ffn_bitserial \
  --ffn_msb_noise_sigma_lsb 0.0

# Learning-CUDA

本项目为 2026 年夏季 InfiniTensor 大模型与人工智能系统训练营 CUDA
方向专业阶段作业，完成了 RMSNorm 与 Flash Attention 两个算子，并提供
NVIDIA、Iluvatar CoreX、MetaX 和 Moore Threads 平台的适配代码。

## 完成情况

| 算子 | float | half | causal | GQA |
| --- | --- | --- | --- | --- |
| RMSNorm | 支持 | 支持 | 不适用 | 不适用 |
| Flash Attention | 支持 | 支持 | 支持 | 支持 |

NVIDIA 平台已在 GeForce RTX 5070 上完成全量验证：

- RMSNorm：26/26 Passed
- Flash Attention：28/28 Passed
- 总计：54/54 Passed

## 项目结构

```text
Learning-CUDA/
├── Makefile
├── LICENSE
├── README.md
├── src
│   ├── kernels.cu      # NVIDIA / Iluvatar CoreX
│   ├── kernels.maca    # MetaX
│   └── kernels.mu      # Moore Threads
└── tester
    ├── tester_iluvatar.o
    ├── tester_metax.o
    ├── tester_moore.o
    ├── tester_nv.o
    └── utils.h
```

## 算子实现

### RMSNorm

对输入矩阵的每一行独立计算：

```text
mean_square = sum_j input[i, j]^2 / hidden_dim
output[i, j] = input[i, j] * rsqrt(mean_square + eps) * weight[j]
```

实现特点：

- 每个线程块负责一行，行之间可以并行执行；
- 使用 FP32 累加平方和，提高 float/half 输入下的数值稳定性；
- NVIDIA 路径使用 warp shuffle 与共享内存完成分层归约；
- 对 float 使用 `float4`、对 half 使用 `half2` 向量化访存，并为非对齐
  或非整除维度保留标量路径；
- 根据数据规模和设备 occupancy 选择线程块大小；
- 复用设备内存，减少多轮 profile 中的分配开销；
- 国产平台路径使用共享内存树形归约，避免依赖特定 warp 宽度。

### Flash Attention

实现行为与接口要求的 scaled dot-product attention 一致：

```text
scores = Q @ K^T / sqrt(head_dim)
probabilities = softmax(scores + causal_mask)
output = probabilities @ V
```

实现特点：

- 支持 float 和 half；
- 支持 causal masking，采用左上对齐的下三角可见区域；
- 支持 GQA，通过 query head 到 KV head 的分组映射复用 K/V；
- 每个线程块负责一个 query/head 输出向量；
- Q 和当前 query 的 scores 保存在共享内存，不生成完整的
  `[target_seq_len, src_seq_len]` 中间矩阵；
- source 位置并行计算 QK，head dimension 按升序使用 FMA 累加；
- softmax 使用减最大值的稳定实现，并按 source 顺序累计分母；
- 输出概率先归一化，再通过 FMA 累加 Value；
- 针对严格的 float causal D=32 用例保留 score 重算路径，以匹配参考实现
  的浮点运算顺序；
- Host 侧缓存 Q/K/V/O 设备缓冲区，减少重复 `malloc/free`。

## 国产平台适配

### Iluvatar CoreX

使用 `src/kernels.cu`，通过 CUDA 兼容接口编译。需要在训练营提供的
BI-150/CoreX 环境中执行最终编译和测试。

### MetaX

`src/kernels.maca` 中包含完整的 RMSNorm 与 Flash Attention 实现，使用
`mcMalloc`、`mcMemcpy`、`mcFree` 等 MACA Runtime API。

### Moore Threads

`src/kernels.mu` 中包含独立完整实现，使用 `musaMalloc`、`musaMemcpy`、
`musaFree` 等 MUSA Runtime API，并保持 C++11 兼容。

MetaX 与 Moore Threads 代码均完成了兼容编译检查，并在 CUDA API 映射环境
下通过 54/54 功能回归；由于当前没有对应国产 GPU，仍需在真实设备环境中
执行最终验证。兼容层结果不等同于国产设备实测结果。

## 环境要求

### NVIDIA

- CUDA Toolkit 11.0 或更高版本；
- 支持 C++17 的编译环境；
- GNU Make。

### Iluvatar CoreX

- 训练营提供的 BI-150/CoreX 环境，或兼容的 CoreX SDK；
- C++17。

### MetaX

- MetaX MACA SDK；
- `mxcc` 编译器；
- C++17。

### Moore Threads

- MUSA SDK；
- `mcc` 编译器；
- C++11。

## 编译与测试

命令均在项目根目录执行。

### NVIDIA

```bash
make clean PLATFORM=nvidia
make PLATFORM=nvidia VERBOSE=true
```

不指定平台时默认使用 NVIDIA：

```bash
make clean
make VERBOSE=true
```

### Iluvatar CoreX

```bash
make clean PLATFORM=iluvatar
make PLATFORM=iluvatar VERBOSE=true
```

### MetaX

```bash
make clean PLATFORM=metax
make PLATFORM=metax VERBOSE=true
```

### Moore Threads

```bash
make clean PLATFORM=moore
make PLATFORM=moore VERBOSE=true
```

### 只测试一个算子

只测试 RMSNorm：

```bash
SKIP_ATTENTION=1 make PLATFORM=nvidia VERBOSE=true
```

只测试 Flash Attention：

```bash
SKIP_RMS_NORM=1 make PLATFORM=nvidia VERBOSE=true
```

## NVIDIA 测试记录

测试设备：NVIDIA GeForce RTX 5070。

最终全量测试结果：

```text
RMSNorm:        26 / 26 Passed
FlashAttention: 28 / 28 Passed
Total:          54 / 54 Passed
```

部分较大 Attention 用例耗时：

```text
Case #6  float: 约 1.06 ms
Case #13 float: 约 3.87 ms
Case #14 float: 约 34.08 ms
Case #14 half:  约 16.98 ms
```

测试时间会受到设备频率、温度、驱动版本和系统负载影响。

## 提交说明

提交作业时填写个人 Fork 仓库地址和最新 commit 链接，无需提交 PR。
评分以截止时间前仓库中的最新提交为准。

# TensorRT Attention 算子测试

基于 NVIDIA TensorRT 文档 Attention 算子示例，测试多头注意力机制（Multi-Head Attention）的构建与推理验证。

## 概述

本项目实现了两种 Attention 算子构建方式：
- **TensorRT >= 10.0**: 使用原生 `add_attention` API
- **TensorRT < 10.0**: 使用 `MatrixMultiply + Softmax` 手动实现等效逻辑

## 环境要求

- Python 3.10+
- TensorRT 10.x (或 8.x)
- NumPy

> 注意：本项目无需 PyCUDA，使用 trtexec 进行推理验证

## 项目文件

```
ch09/
├── test_attention.py      # 主测试脚本（生成引擎和输入）
├── compare.py             # 对比 TRT 输出与 NumPy 参考结果
├── run_trtexec.sh         # trtexec 推理脚本
├── attention_engine.trt   # 生成的 TensorRT 引擎
├── attention_inputs/       # 输入数据目录
│   ├── query.bin
│   ├── key.bin
│   ├── value.bin
│   └── mask.bin
└── output.json            # trtexec 推理输出
```

## 使用方法

### 1. 构建引擎

```bash
python3 test_attention.py
```

输出：
- `attention_engine.trt` - TensorRT 引擎文件
- `attention_inputs/` - 输入数据目录
- trtexec 推理命令

### 2. 运行推理

将 `run_trtexec.sh` 复制到 TensorRT Docker 环境中执行：

```bash
# 方式 1: 使用 Docker
docker exec <container_id> bash /workspace/TensorRT/OpenMMLab_test/ch09/run_trtexec.sh

# 方式 2: 直接在容器中运行
cd /workspace/TensorRT/OpenMMLab_test/ch09
./run_trtexec.sh
```

### 3. 对比结果

```bash
python3 compare.py
```

输出 TensorRT 输出与 NumPy 参考结果的对比：

```
=== 对比结果 ===
最大绝对误差: 0.000001
平均绝对误差: 0.000000
误差 < 1e-3: 128 / 128 (100.0%)
```

## Attention 算子说明

### 输入

| 输入 | 形状 | 类型 | 说明 |
|------|------|------|------|
| Query | (B, H, S_q, D) | float16 | 查询张量 |
| Key | (B, H, S_kv, D) | float16 | 键张量 |
| Value | (B, H, S_kv, D) | float16 | 值张量 |
| Mask | (B, 1, S_q, S_kv) | bool/float16 | 注意力掩码 |

其中：
- B: batch size
- H: num heads
- S_q: query 序列长度
- S_kv: key/value 序列长度
- D: head dimension

### 输出

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(D) + mask) @ V
```

形状: (B, H, S_q, D)

## 相关文档

- [TensorRT Multi-Attention Fusion](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-transformers.html#multi-head-attention-fusion)
- [TensorRT IAttention API](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python_api_new.html)

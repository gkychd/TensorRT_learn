#!/bin/bash

# 1. 测试 V10 原生 API 引擎 (假设你已经用 v10 脚本生成了引擎)
echo "=== 正在测试 TensorRT 10.0 原生 Attention API ==="
trtexec --loadEngine=attention_engine.trt \
        --duration=10 \
        --useCudaGraph \
        --noDataTransfers \
        --verbose > v10_profile.log 2>&1

# 2. 提取关键性能指标
V10_LATENCY=$(grep "GPU Compute Time" v10_profile.log | tail -n 1)
echo "V10 性能: $V10_LATENCY"

# 3. 观察 Kernel 融合情况 (搜索是否存在融合后的计算节点)
echo "检查 Kernel 融合情况..."
grep -E "Myelin|FusedAttention|fmha" v10_profile.log | head -n 5
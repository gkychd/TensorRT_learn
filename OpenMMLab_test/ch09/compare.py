#!/usr/bin/env python3
"""
对比 TensorRT 输出与 NumPy 参考结果
"""
import numpy as np
import json
import os

# 设置随机种子以复现数据
np.random.seed(42)

# 读取输入数据
q_shape = (1, 8, 1, 16)
kv_shape = (1, 8, 200, 16)
mask_shape = (1, 1, 1, 200)

# 加载输入数据
query = np.fromfile('attention_inputs/query.bin', dtype=np.float16).reshape(q_shape)
key = np.fromfile('attention_inputs/key.bin', dtype=np.float16).reshape(kv_shape)
value = np.fromfile('attention_inputs/value.bin', dtype=np.float16).reshape(kv_shape)

print("=== 输入数据 ===")
print(f"Query shape: {query.shape}, dtype: {query.dtype}")
print(f"Key shape: {key.shape}, dtype: {key.dtype}")
print(f"Value shape: {value.shape}, dtype: {value.dtype}")

# NumPy 参考实现
def numpy_reference_attention(query, key, value):
    """NumPy 参考实现（支持 Q/KV seq_len 不同）"""
    q = query.astype(np.float32)  # (B, H, S_q, D)
    k = key.astype(np.float32)    # (B, H, S_kv, D)
    v = value.astype(np.float32)  # (B, H, S_kv, D)
    d = q.shape[-1]
    # Q @ K^T / sqrt(d) -> (B, H, S_q, S_kv)
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d)
    # softmax over S_kv dim
    scores_exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn_weights = scores_exp / scores_exp.sum(axis=-1, keepdims=True)
    # attn_weights @ V -> (B, H, S_q, D)
    out = np.matmul(attn_weights, v)
    return out.astype(np.float32)

np_ref = numpy_reference_attention(query, key, value)
print(f"\n=== NumPy 参考输出 ===")
print(f"Shape: {np_ref.shape}, dtype: {np_ref.dtype}")
print(f"First 10 values: {np_ref.flatten()[:10]}")

# 读取 TensorRT 输出
with open('output.json', 'r') as f:
    trt_output = json.load(f)

trt_data = np.array(trt_output[0]['values'], dtype=np.float32)
trt_data = trt_data.reshape(q_shape)  # (1, 8, 1, 16)

print(f"\n=== TensorRT 输出 ===")
print(f"Shape: {trt_data.shape}, dtype: {trt_data.dtype}")
print(f"First 10 values: {trt_data.flatten()[:10]}")

# 对比
diff = np.abs(trt_data - np_ref)
max_diff = np.max(diff)
mean_diff = np.mean(diff)

print(f"\n=== 对比结果 ===")
print(f"最大绝对误差: {max_diff:.6f}")
print(f"平均绝对误差: {mean_diff:.6f}")
print(f"误差 < 1e-3: {np.sum(diff < 1e-3)} / {diff.size} ({100*np.sum(diff < 1e-3)/diff.size:.1f}%)")
print(f"误差 < 1e-2: {np.sum(diff < 1e-2)} / {diff.size} ({100*np.sum(diff < 1e-2)/diff.size:.1f}%)")
print(f"误差 < 1e-1: {np.sum(diff < 1e-1)} / {diff.size} ({100*np.sum(diff < 1e-1)/diff.size:.1f}%)")

# 显示每 head 的误差
print(f"\n=== 每个 Head 的最大误差 ===")
for i in range(q_shape[1]):
    head_diff = np.max(np.abs(trt_data[0, i, 0, :] - np_ref[0, i, 0, :]))
    print(f"  Head {i}: {head_diff:.6f}")

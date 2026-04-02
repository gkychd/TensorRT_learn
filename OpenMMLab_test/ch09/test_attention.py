#!/usr/bin/env python3
"""
TensorRT Attention Operator 完整测试脚本 (无 PyCUDA, 无 PyTorch 版本)

基于 NVIDIA TensorRT 文档 Attention 算子示例:
https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/operators/Attention.html#examples

支持两种模式:
  - TensorRT >= 10.0: 使用原生 add_attention API
  - TensorRT  < 10.0: 使用 MatrixMultiply + Softmax 手动实现等效逻辑
"""

import sys
import os
import numpy as np

try:
    import tensorrt as trt
except ImportError:
    print("错误: 未安装 tensorrt，请先安装 TensorRT Python 包。")
    sys.exit(1)

TRT_VERSION = tuple(int(x) for x in trt.__version__.split(".")[:2])
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_attention_network_v8(builder: trt.Builder, q_shape, kv_shape, mask_shape):
    """
    TensorRT < 10.0: 用 MatrixMultiply + Softmax 手动实现 Scaled Dot-Product Attention

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(h) + mask) @ V
    输入形状: [b, d, s, h]  (batch, num_heads, seq_len, head_dim)
    mask 为 additive mask，0 表示允许 attend，-inf 表示屏蔽
    """
    print("使用手动 build_attention_network_v8 ...")
    network = builder.create_network(
        flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    b, d, s_q, h = kv_shape
    query = network.add_input("query", dtype=trt.float16, shape=q_shape)
    key = network.add_input("key", dtype=trt.float16, shape=kv_shape)
    value = network.add_input("value", dtype=trt.float16, shape=kv_shape)
    mask = network.add_input("mask", dtype=trt.float16, shape=mask_shape)

    # Q @ K^T => [b, d, s_q, s_kv]
    bmm1 = network.add_matrix_multiply(
        query, trt.MatrixOperation.NONE,
        key, trt.MatrixOperation.TRANSPOSE,
    )
    bmm1.name = "BMM1_QK"

    # scale = 1 / sqrt(head_dim)
    scale_val = np.array([1.0 / np.sqrt(h)], dtype=np.float16)
    print(f"scale_val {scale_val}")
    scale_const = network.add_constant((1, 1, 1, 1), scale_val)
    scale_layer = network.add_elementwise(
        bmm1.get_output(0), scale_const.get_output(0), trt.ElementWiseOperation.PROD
    )
    scale_layer.name = "Scale"

    # additive mask (0.0 = attend, -inf = block)
    mask_add = network.add_elementwise(
        scale_layer.get_output(0), mask, trt.ElementWiseOperation.SUM
    )
    mask_add.name = "MaskAdd"

    # softmax over last dimension (s_kv)
    softmax = network.add_softmax(mask_add.get_output(0))
    softmax.axes = 1 << 3  # axis=3 (s_kv dimension)
    softmax.name = "Softmax"

    # Attention @ V => [b, d, s_q, h]
    bmm2 = network.add_matrix_multiply(
        softmax.get_output(0), trt.MatrixOperation.NONE,
        value, trt.MatrixOperation.NONE,
    )
    bmm2.name = "BMM2_AV"

    network.mark_output(bmm2.get_output(0))
    return network

def build_attention_network_v10(builder: trt.Builder, q_shape, kv_shape, mask_shape):
    """TensorRT >= 10.0: 使用原生 add_attention API，支持 Q/KV shape 不同"""
    print("使用原生 build_attention_network_v10 ...")
    network = builder.create_network(
        flags=1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    query = network.add_input("query", dtype=trt.float16, shape=q_shape)
    key = network.add_input("key", dtype=trt.float16, shape=kv_shape)
    value = network.add_input("value", dtype=trt.float16, shape=kv_shape)
    mask = network.add_input("mask", dtype=trt.bool, shape=mask_shape)
    
    layer = network.add_attention(
        query, key, value, trt.AttentionNormalizationOp.SOFTMAX, False
    )
    layer.mask = mask
    network.mark_output(layer.get_output(0))
    return network

def build_attention_network_v10_fixed(builder: trt.Builder, q_shape, kv_shape, mask_shape):
    """修正后的 TensorRT >= 10.0 实现"""
    print("使用修正后的 build_attention_network_v10 ...")
    
    # 注意：STRONGLY_TYPED 是 TRT 10 的新特性，确保版本支持
    network = builder.create_network(
        flags=1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    
    # 1. 定义输入 (为了兼容，这里假设外部传入的 mask 依然是 float16 的 additive mask)
    # 如果外部能改，建议直接传 bool mask
    query = network.add_input("query", dtype=trt.float16, shape=q_shape)
    key = network.add_input("key", dtype=trt.float16, shape=kv_shape)
    value = network.add_input("value", dtype=trt.float16, shape=kv_shape)
    
    # --- 修正点 1: 手动添加缩放 ---
    b, d, s_kv, h = kv_shape
    # 计算 scale = 1 / sqrt(h)
    scale_val = np.array([1.0 / np.sqrt(h)], dtype=np.float16)
    scale_const = network.add_constant((1, 1, 1, 1), scale_val)
    
    # 将 query 乘以 scale (或者乘 key 也可以)
    scaled_query = network.add_elementwise(
        query, scale_const.get_output(0), trt.ElementWiseOperation.PROD
    )
    
    # --- 修正点 2: 处理 Mask ---
    # 原生 API 的 layer.mask 需要 bool 类型。
    # 如果你的输入 mask 是 float16 (-inf/0)，你需要先把它转成 bool。
    # 这里假设我们修改了输入定义，或者在外部处理好 bool mask 再传入。
    # 为了演示，这里假设我们有一个 bool 类型的 mask 输入
    mask = network.add_input("mask", dtype=trt.bool, shape=mask_shape)

    # 创建 Attention 层
    layer = network.add_attention(
        scaled_query.get_output(0), # 使用缩放后的 Q
        key, 
        value, 
        trt.AttentionNormalizationOp.SOFTMAX, 
        False
    )
    
    # 赋值布尔掩码
    layer.mask = mask
    
    network.mark_output(layer.get_output(0))
    return network

def prepare_test_data_from_files(q_shape, kv_shape, mask_shape, use_native_api: bool, use_local_input: bool = True, input_dir: str = "attention_inputs"):
    """生成或读取测试数据

    Args:
        q_shape: Query 张量形状
        kv_shape: Key/Value 张量形状
        mask_shape: Mask 张量形状
        use_native_api: 是否使用原生 API (影响 mask 类型)
        use_local_input: 是否尝试从本地文件读取输入 (默认 True)
        input_dir: 输入文件目录

    Returns:
        inputs: 包含 query, key, value, mask 的字典
    """
    inputs = {}

    # 尝试从本地文件读取
    if use_local_input:
        local_files_exist = True
        for name in ['query', 'key', 'value', 'mask']:
            fpath = os.path.join(input_dir, f"{name}.bin")
            if not os.path.exists(fpath):
                local_files_exist = False
                break

        if local_files_exist:
            print(f"从本地读取输入文件: {input_dir}")
            inputs['query'] = np.fromfile(os.path.join(input_dir, "query.bin"), dtype=np.float16).reshape(q_shape)
            inputs['key'] = np.fromfile(os.path.join(input_dir, "key.bin"), dtype=np.float16).reshape(kv_shape)
            inputs['value'] = np.fromfile(os.path.join(input_dir, "value.bin"), dtype=np.float16).reshape(kv_shape)
            # mask 文件名可能不带 .bin 后缀，尝试不同后缀
            # 注意：V10 原生 API 的 mask 是 bool 类型（保存为 uint8），V8 是 float16
            mask_path = os.path.join(input_dir, "mask.bin")
            if os.path.exists(mask_path):
                # 检查文件大小来判断类型
                file_size = os.path.getsize(mask_path)
                expected_bool = np.prod(mask_shape)       # bool = 1 byte
                if file_size == expected_bool:
                    inputs['mask'] = np.fromfile(mask_path, dtype=np.uint8).reshape(mask_shape)
                else:
                    inputs['mask'] = np.fromfile(mask_path, dtype=np.float16).reshape(mask_shape)
            else:
                mask_path = os.path.join(input_dir, "mask")
                if os.path.exists(mask_path):
                    file_size = os.path.getsize(mask_path)
                    expected_bool = np.prod(mask_shape)
                    if file_size == expected_bool:
                        inputs['mask'] = np.fromfile(mask_path, dtype=np.uint8).reshape(mask_shape)
                    else:
                        inputs['mask'] = np.fromfile(mask_path, dtype=np.float16).reshape(mask_shape)
                else:
                    local_files_exist = False

        if local_files_exist:
            print(f"  成功读取 {len(inputs)} 个输入文件")
            return inputs
        else:
            print(f"本地文件不存在或不全，生成随机数据")

    # 生成随机数据
    print("生成随机测试数据...")
    q_tensor = np.random.randn(*q_shape).astype(np.float16)
    k_tensor = np.random.randn(*kv_shape).astype(np.float16)
    v_tensor = np.random.randn(*kv_shape).astype(np.float16)

    assert q_tensor.shape == q_shape, f"Q shape mismatch: {q_tensor.shape} vs {q_shape}"
    assert k_tensor.shape == kv_shape, f"K shape mismatch: {k_tensor.shape} vs {kv_shape}"
    assert v_tensor.shape == kv_shape, f"V shape mismatch: {v_tensor.shape} vs {kv_shape}"

    inputs = {
        "query": q_tensor,
        "key": k_tensor,
        "value": v_tensor,
    }

    if use_native_api:
        # mask: (B, 1, S_q, S_kv) 或 (B, H, S_q, S_kv)
        # 这里用全 True 表示 attend to all
        mask_data = np.ones(mask_shape, dtype=np.bool_)
    else:
        # 手动实现路径暂不支持，此处仅为占位
        mask_data = np.zeros(mask_shape, dtype=np.float16)

    inputs["mask"] = mask_data
    return inputs

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
    return out.astype(np.float16)

def build_engine(builder, network, engine_path):
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB

    print("正在构建 TensorRT 引擎 ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("引擎构建失败！")

    # 保存引擎文件
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"引擎已保存至: {engine_path}")
    return engine_path

def numpy_reference_attention(query, key, value):
    """NumPy 参考实现，用于对比验证"""
    q = query.astype(np.float32)
    k = key.astype(np.float32)
    v = value.astype(np.float32)
    h = q.shape[-1]
    # Q @ K^T / sqrt(h)
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(h)
    # softmax along last axis
    scores_exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn_weights = scores_exp / scores_exp.sum(axis=-1, keepdims=True)
    out = np.matmul(attn_weights, v)
    return out.astype(np.float16)

def main():
    import os

    print(f"TensorRT 版本: {trt.__version__}")
    print("-" * 60)
    if 0 :
        q_shape = (1, 8, 1, 16)
        kv_shape = (1, 8, 2, 16)
        mask_shape = (1, 1, 1, 2)  # (B, 1, S_q, S_kv) —— 可被广播到 (B, H, S_q, S_kv)
    else :
        q_shape = (1, 8, 1, 16)
        kv_shape = (1, 8, 200, 16)
        mask_shape = (1, 1, 1, 200)  # (B, 1, S_q, S_kv) —— 可被广播到 (B, H, S_q, S_kv)

    use_native = TRT_VERSION >= (10, 0)
    # use_native = True
    #use_native = False

    print("模式: 原生 add_attention API (TRT >= 10.0)")
    print(f"Query 形状: {q_shape}")
    print(f"Key/Value 形状: {kv_shape}")
    print(f"Mask 形状: {mask_shape}  # (B, 1, S_q, S_kv)")
    print("-" * 60)

    # 构建引擎
    builder = trt.Builder(TRT_LOGGER)

    if use_native:
        network = build_attention_network_v10_fixed(builder, q_shape, kv_shape, mask_shape)
    else:
        network = build_attention_network_v8(builder, q_shape, kv_shape, mask_shape)

    engine_path = "attention_engine.trt"
    build_engine(builder, network, engine_path)

    # 生成输入数据并保存为二进制文件
    inputs = prepare_test_data_from_files(q_shape, kv_shape, mask_shape, use_native, True)
    input_dir = "attention_inputs"
    os.makedirs(input_dir, exist_ok=True)

    for name, data in inputs.items():
        file_path = os.path.join(input_dir, f"{name}.bin")
        # V10 原生 API 的 mask 是 bool 类型，需要特殊处理
        if name == "mask" and use_native:
            # 将 bool 转为 uint8 保存 (trtexec 支持)
            data.astype(np.uint8).tofile(file_path)
        else:
            data.astype(np.float16).tofile(file_path)
        print(f"已保存: {file_path}")

    # 输出 NumPy 参考结果
    np_ref = numpy_reference_attention(inputs["query"], inputs["key"], inputs["value"])
    print("\n--- NumPy 参考输出 (每个 head) ---")
    for i in range(q_shape[1]):
        print(f"  head {i}: {np_ref[0, i, 0, :3]}...")

    # 生成 trtexec 命令
    if use_native:
        input_shapes = f"--optShape=query:{q_shape},key:{kv_shape},value:{kv_shape},mask:{mask_shape}"
    else:
        input_shapes = f"--optShape=query:{q_shape},key:{kv_shape},value:{kv_shape},mask:{mask_shape}"

    trtexec_cmd = f"""
# ============================================================
# 推理验证命令（使用 trtexec）
# ============================================================

trtexec --engine={engine_path} \\
    --inputFile={input_dir} \\
    {input_shapes} \\
    --dumpOutput \\
    --output=output
""".strip()

    print("\n" + "=" * 60)
    print("构建完成！")
    print("=" * 60)
    print(f"\n引擎文件: {engine_path}")
    print(f"输入目录: {input_dir}")
    print(f"\n使用 trtexec 推理验证:\n")
    print(trtexec_cmd)
    print("\n请在 GPU 环境中运行 trtexec 命令进行推理验证。")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    sys.exit(main())
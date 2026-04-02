#!/usr/bin/env python3
"""
使用 TensorRT Python API 进行推理（无 pycuda 版本）
需要 tensorrt 10+ 的 cuda_malloc 支持
"""
import tensorrt as trt
import numpy as np
import ctypes

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 加载引擎
with open('attention_engine.trt', 'rb') as f:
    engine_data = f.read()

print(f"引擎文件大小: {len(engine_data)} bytes")
print(f"文件头部: {engine_data[:20]}")

# 尝试反序列化
try:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)
    print(f"\n引擎加载成功!")
    print(f"输入输出张量数量: {engine.num_io_tensors}")

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        print(f"  {name}: shape={shape}, dtype={dtype}, mode={mode}")

    # 创建推理上下文
    context = engine.create_execution_context()
    print("\n推理上下文创建成功!")

except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    traceback.print_exc()

#!/usr/bin/env python3
import tensorrt as trt
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 加载引擎
with open('attention_engine.trt', 'rb') as f:
    engine_data = f.read()

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_data)
context = engine.create_execution_context()

# 加载输入数据
inputs = {}
for name in ['query', 'key', 'value', 'mask']:
    data = np.fromfile(f'attention_inputs/{name}.bin', dtype=np.float16)
    inputs[name] = data
    print(f"Loaded {name}: {data.shape}, first 5: {data[:5]}")

# 设置输入形状
input_shapes = {
    'query': (1, 8, 1, 16),
    'key': (1, 8, 200, 16),
    'value': (1, 8, 200, 16),
    'mask': (1, 1, 1, 200)
}

# 设置张量地址并推理
for name, shape in input_shapes.items():
    context.set_input_shape(name, shape)

# 分配输出内存
output_name = engine.get_tensor_name(0)  # 假设第一个是输出
output_shape = engine.get_tensor_shape(output_name)
output_size = np.prod(output_shape) * 2  # float16 = 2 bytes

# 使用 TensorRT 10 的新 API
d_output = np.empty(output_shape, dtype=np.float16)

# 尝试使用 execute_v2
# 需要分配 GPU 内存，这里用 numpy 来验证结果
print(f"\nOutput shape: {output_shape}")

# 使用 CPU 模拟推理（验证网络结构）
print("\n网络层信息:")
for i in range(engine.num_layers):
    layer = engine.get_layer(i)
    print(f"  Layer {i}: {layer.name} -> {layer.type}")

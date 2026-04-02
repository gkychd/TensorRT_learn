#!/usr/bin/env python3
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

with open('attention_engine.trt', 'rb') as f:
    engine_data = f.read()

print(f'Engine file size: {len(engine_data)} bytes')
print(f'First 20 bytes: {engine_data[:20]}')

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_data)
print(f'Engine loaded successfully!')
print(f'Number of tensors: {engine.num_io_tensors}')
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = engine.get_tensor_shape(name)
    dtype = engine.get_tensor_dtype(name)
    mode = engine.get_tensor_mode(name)
    print(f'  {name}: {shape} {dtype} {mode}')

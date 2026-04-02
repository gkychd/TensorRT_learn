#!/usr/bin/env python3
"""
生成 ONNX 模型用于验证 Attention 算子
"""
import numpy as np
import onnx
from onnx import helper, TensorProto

# 创建 ONNX 图
# 输入
query = helper.make_tensor_value_info('query', TensorProto.FLOAT, [1, 8, 1, 16])
key = helper.make_tensor_value_info('key', TensorProto.FLOAT, [1, 8, 200, 16])
value = helper.make_tensor_value_info('value', TensorProto.FLOAT, [1, 8, 200, 16])
mask = helper.make_tensor_value_info('mask', TensorProto.FLOAT, [1, 1, 1, 200])

# 输出
output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 8, 1, 16])

# 创建初始值 (scale = 1/sqrt(16))
scale_value = np.array([1.0 / np.sqrt(16.0)], dtype=np.float32)
scale_init = helper.make_tensor('scale', TensorProto.FLOAT, [1, 1, 1, 1], scale_value.tobytes(), raw=True)

# 节点: Q @ K^T
matmul_qk = helper.make_node('MatMul', ['query', 'key'], ['qk'], name='MatMul_QK')

# 节点: scale
scale_node = helper.make_node('Mul', ['qk', 'scale'], ['scaled'], name='Scale')

# 节点: add mask
mask_add = helper.make_node('Add', ['scaled', 'mask'], ['masked'], name='MaskAdd')

# 节点: softmax
softmax = helper.make_node('Softmax', ['masked'], ['attn_weights'], axis=3, name='Softmax')

# 节点: attn @ V
output_node = helper.make_node('MatMul', ['attn_weights', 'value'], ['output'], name='MatMul_AV')

# 创建图
graph = helper.make_graph([matmul_qk, scale_node, mask_add, softmax, output_node],
                          'Attention',
                          [query, key, value, mask],
                          [output],
                          [scale_init])

# 创建模型
model = helper.make_model(graph, producer_name='attention_test')
model.opset_import[0].version = 13

# 保存
onnx.save(model, 'attention.onnx')
print("ONNX 模型已保存: attention.onnx")
print("\n使用 onnxruntime 验证:")
print("  python3 -c \"import onnxruntime as ort; sess = ort.InferenceSession('attention.onnx'); print(sess.get_inputs())\"")

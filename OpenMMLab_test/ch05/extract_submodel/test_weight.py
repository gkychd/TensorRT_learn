import onnx
import numpy as np
from onnx import numpy_helper

# 加载模型
model_path = 'whole_model.onnx'
try:
    model_onnx = onnx.load(model_path)
except FileNotFoundError:
    print(f"错误：找不到文件 '{model_path}'，请确认文件已生成。")
    exit()

# 检查 initializer (这就是权重和偏置存放的地方)
initializers = model_onnx.graph.initializer
print(f"包含的权重/常量数量: {len(initializers)}")
print("-" * 60)

if len(initializers) == 0:
    print("提示：该模型没有包含任何权重 (initializer 为空)。")
else:
    # 打印前 5 个权重的详细信息
    for i, init in enumerate(initializers):
        if i >= 5:
            print(f"... (还有 {len(initializers) - 5} 个权重未显示)")
            break
        
        # 1. 将 ONNX 的 TensorProto 转换为 numpy 数组
        weight_array = numpy_helper.to_array(init)
        
        # 2. 打印基本信息
        print(f"[{i}] 名称: {init.name}")
        print(f"    形状 (Shape): {weight_array.shape}")
        print(f"    数据类型: {weight_array.dtype}")
        
        # 3. 打印统计特征 (快速判断是否为随机初始化)
        # 随机初始化的权重通常均值接近 0，且有正有负
        print(f"    统计信息: Min={weight_array.min():.4f}, Max={weight_array.max():.4f}, Mean={weight_array.mean():.4f}")
        
        # 4. 打印具体数值预览 (只打印展平后的前 8 个数)
        flat_data = weight_array.flatten()
        preview = flat_data[:8] # 取前8个
        values_str = ", ".join([f"{v:.4f}" for v in preview])
        if len(flat_data) > 8:
            values_str += ", ..."
            
        print(f"    数值预览: [{values_str}]")
        print("-" * 60)

# 检查节点数量 (验证结构是否完整)
print(f"\n包含的算子节点数量 (NodeProto): {len(model_onnx.graph.node)}")

# (可选) 简单分析一下算子类型，看看结构对不对
op_counts = {}
for node in model_onnx.graph.node:
    op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

print("\n算子类型分布:")
for op_type, count in op_counts.items():
    print(f"  - {op_type}: {count} 个")
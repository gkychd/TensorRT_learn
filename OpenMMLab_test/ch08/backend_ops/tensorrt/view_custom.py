#!/usr/bin/env python3
import subprocess
import os
import re

# 方法1: 使用 nm 从 .so 文件中提取插件名称
so_file = "./libmmdeploy_tensorrt_ops.so"

if not os.path.exists(so_file):
    print(f"Error: {so_file} not found!")
    exit(1)

print(f"=== Extracting plugins from {so_file} ===\n")

# 使用 nm 提取 getPluginName 符号
result = subprocess.run(
    ["nm", so_file],
    capture_output=True,
    text=True
)

# 解析符号
plugins = set()
for line in result.stdout.splitlines():
    if "getPluginName" in line and "Creator" in line:
        # 提取 demangled 名称
        parts = line.split()
        if len(parts) >= 3:
            mangled_name = parts[2]

            # 使用 c++filt 进行 demangle
            demangle_result = subprocess.run(
                ["c++filt", mangled_name],
                capture_output=True,
                text=True
            )
            demangled = demangle_result.stdout.strip()

            # 提取 Creator 名称
            # 格式: mmdeploy::TRTBicubicInterpolateCreator::getPluginName() const
            match = re.search(r'::(\w+Creator)::', demangled)
            if match:
                creator_name = match.group(1)
                # 去掉 Creator 后缀得到插件名
                plugin_name = creator_name.replace('Creator', '')
                plugins.add(plugin_name)

# 打印结果
print("Registered custom operators:")
for p in sorted(plugins):
    print(f"  - {p}")

print(f"\nTotal: {len(plugins)} plugins")

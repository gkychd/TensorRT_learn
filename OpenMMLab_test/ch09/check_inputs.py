#!/usr/bin/env python3
import numpy as np
import os

# 检查输入文件
input_dir = "attention_inputs"
for fname in os.listdir(input_dir):
    fpath = os.path.join(input_dir, fname)
    data = np.fromfile(fpath, dtype=np.float16)
    print(f"{fname}: {data.shape}, dtype={data.dtype}, first 5 values: {data[:5]}")

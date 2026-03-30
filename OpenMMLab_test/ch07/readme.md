### 1 c++_build

```
mkdir build && cd build
cmake ..
make -j
```

1. 直接通过trt构建engine
2. 通过读取onnx构建engine
3. tensorrt模型推理
4. 比较两种方式构建模型的推理结果

### 2 python_build

1. 直接通过trt构建engine
2. 生成onnx （在有torch和onnx库的环境下使用)
3. 由于tensorrt的docker环境下没有torch和onnx，其他功能暂位未用py实现

```bash
cd /workspace/TensorRT/OpenMMLab_test/ch08/srcnn_converter
mkdir -p build && cd build
cmake .. -DTRT_INCLUDE=/workspace/TensorRT/include -DTRT_LIB=/usr/lib/x86_64-linux-gnu
make

# 设置环境变量并运行
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
./srcnn_converter --onnx=../srcnn3.onnx --engine=srcnn3.engine --plugin_lib=.../../backend_ops/tensorrt/build/libmmdeploy_tensorrt_ops.so
```

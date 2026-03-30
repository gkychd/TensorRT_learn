1. 编译

```
cd /TensorRT/OpenMMLab_test/ch08/backend_ops/tensorrt
mkdir -p build && cd build
cmake .. -DTRT_INCLUDE=/workspace/TensorRT/include -DTRT_LIB=/usr/lib64
make -j
```

2. 查看算子库中包含的算子

方法1：

```bash
nm libmmdeploy_tensorrt_ops.so | grep "getPluginName" | c++filt
```

方法2：

```python
import ctypes
import tensorrt as trt

# 创建 logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

print("Loading custom plugin library...")
lib = ctypes.CDLL("./libmmdeploy_tensorrt_ops.so", ctypes.RTLD_GLOBAL)
print(f"Library loaded: {lib}")

# 初始化插件
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

plugin_registry = trt.get_plugin_registry()

# 正确的插件名称列表
print("\n=== Checking all plugins ===")
all_plugins = [
    ("TRTBicubicInterpolate", "1"),
    ("TRTBatchedNMS", "1"),
    ("TRTBatchedRotatedNMS", "1"),
    ("GatherTopk", "1"),
    ("GridPriorsTRT", "1"),
    ("grid_sampler", "1"),           # 注意是小写
    ("MMCVRoiAlign", "1"),
    ("MMCVMultiLevelRoiAlign", "1"),
    ("MMCVMultiLevelRotatedRoiAlign", "1"),
    ("MMCVMultiScaleDeformableAttention", "1"),
    ("ScatterND", "1"),
]

found = []
for name, version in all_plugins:
    try:
        creator = plugin_registry.get_plugin_creator(name, version, "")
        if creator:
            found.append(creator.name)
            print(f"  Found: {creator.name} (version {creator.plugin_version})")
    except:
        pass

print(f"\n=== Total: {len(found)} custom operators ===")

```

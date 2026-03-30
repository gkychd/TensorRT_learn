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
    ("DynamicTRTResize", "1"),
]

found = []
for name, version in all_plugins:
    # 尝试不同的版本号
    for v in [version, "1", "2", "3", ""]:
        try:
            creator = plugin_registry.get_plugin_creator(name, v, "")
            if creator:
                found.append(creator.name)
                print(f"  Found: {creator.name} (version {creator.plugin_version})")
                break
        except:
            pass

print(f"\n=== Total: {len(found)} custom operators ===")

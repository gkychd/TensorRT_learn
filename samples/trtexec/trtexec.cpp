/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// trtexec.cpp - TensorRT 命令行推理工具的主入口文件
// 功能: 提供从 ONNX/UFF/caffemodel 构建 TensorRT Engine 并执行推理的命令行工具
// 用法: trtexec --onnx=model.onnx --saveEngine=engine.trt

#include <algorithm>      // 标准算法库 (sort, find等)
#include <cctype>         // 字符处理函数
#include <chrono>         // 时间相关功能,用于性能计时
#include <cmath>          // 数学函数
#include <functional>     // 函数对象和std::function
#include <iostream>        // 输入输出流
#include <memory>         // 智能指针 (unique_ptr, shared_ptr)
#include <sys/stat.h>     // 文件状态查询 (用于检查文件是否存在)
#include <vector>         // 动态数组容器

// TensorRT 核心头文件
#include "NvInfer.h"          // TensorRT 主API,包含IBuilder, IRuntime等核心类
#include "NvInferPlugin.h"   // TensorRT 插件系统头文件

// TensorRT Samples 公共组件
#include "buffers.h"          // 输入输出缓冲区管理 (用于GPU内存分配/拷贝)
#include "common.h"           // 通用工具函数和类
#include "logger.h"           // 日志系统封装
#include "sampleDevice.h"     // CUDA设备管理 (设备选择,查询等)
#include "sampleEngines.h"    // Engine构建相关 (getEngineBuildEnv等)
#include "sampleInference.h" // 推理执行相关 (runInference等)
#include "sampleOptions.h"    // 命令行选项解析 (AllOptions类)
#include "sampleReporting.h" // 性能报告输出 (printPerformanceReport等)

// 使用命名空间简化代码
using namespace nvinfer1;       // TensorRT 1.x 版本主命名空间 (核心API)
using namespace sample;         // Samples 工具类命名空间
using namespace samplesCommon;  // Samples 公共组件命名空间

#if ENABLE_UNIFIED_BUILDER
// 统一Builder模式 (TensorRT 10+ 新特性)
using namespace nvinfer2::safe; // 安全TensorRT命名空间

// 全局安全记录器 - 用于记录安全模式下的构建/推理信息
// __attribute__((weak)) 允许用户自定义实现来覆盖默认行为
__attribute__((weak)) std::shared_ptr<sample::SampleSafeRecorder> gSafeRecorder
    = std::make_shared<sample::SampleSafeRecorder>(nvinfer2::safe::Severity::kINFO);
#endif // ENABLE_UNIFIED_BUILDER

namespace
{ // 匿名命名空间 - 仅在本文件内可见,类似于static关键字

// 智能指针类型别名 - 管理动态库的加载和卸载
using LibraryPtr = std::unique_ptr<DynamicLibrary>;

// 函数指针变量 - 保存TensorRT内部API函数地址
// 这些是TensorRT的内部API,用于创建核心对象
std::function<void*(void*, int32_t)> pCreateInferRuntimeInternal{};          // 创建IRuntime
std::function<void*(void*, void*, int32_t)> pCreateInferRefitterInternal{};   // 创建IRefitter
std::function<void*(void*, int32_t)> pCreateInferBuilderInternal{};           // 创建IBuilder
std::function<void*(void*, void*, int)> pCreateNvOnnxParserInternal{};       // 创建ONNX Parser
std::function<void*(void*, void*, int)> pCreateNvOnnxRefitterInternal{};     // 创建ONNX Refitter

//! 跟踪trtexec使用的运行时模式
//! 必须使用全局变量,因为库初始化函数的API组织方式需要
// kFULL: 完整版TensorRT,包含所有功能
// kLEANT: 精简版,只支持推理
// kSAFE: 安全版,用于安全推理场景
RuntimeMode gUseRuntime = RuntimeMode::kFULL;

// 初始化TensorRT核心库 (libnvinfer.so)
// @return: 库加载成功返回true,否则返回false
bool initNvinfer()
{
#if !TRT_STATIC
    // 动态链接模式: 从共享库加载
    static LibraryPtr libnvinferPtr{};  // 静态变量,保证只加载一次
    // fetchPtrs: 加载成功后获取函数指针的回调
    auto fetchPtrs = [](DynamicLibrary* l) {
        // 获取 createInferRuntime_INTERNAL 函数地址
        pCreateInferRuntimeInternal = l->symbolAddress<void*(void*, int32_t)>("createInferRuntime_INTERNAL");
        try
        {
            // 获取 createInferRefitter_INTERNAL 函数地址 (可能不存在)
            pCreateInferRefitterInternal
                = l->symbolAddress<void*(void*, void*, int32_t)>("createInferRefitter_INTERNAL");
        }
        catch (const std::exception& e)
        {
            sample::gLogWarning << "Could not load function createInferRefitter_INTERNAL : " << e.what() << std::endl;
        }

        // 只有完整版才需要Builder
        if (gUseRuntime == RuntimeMode::kFULL)
        {
            pCreateInferBuilderInternal = l->symbolAddress<void*(void*, int32_t)>("createInferBuilder_INTERNAL");
        }
    };
    // 调用通用库加载函数
    return initLibrary(libnvinferPtr, getRuntimeLibraryName(gUseRuntime), fetchPtrs);
#else
    // 静态链接模式: 直接使用链接进来的函数
    pCreateInferRuntimeInternal = createInferRuntime_INTERNAL;
    pCreateInferRefitterInternal = createInferRefitter_INTERNAL;
    pCreateInferBuilderInternal = createInferBuilder_INTERNAL;
    return true;
#endif // !TRT_STATIC
}

// 初始化ONNX Parser库 (libnvonnxparser.so)
// @return: 库加载成功返回true,否则返回false
bool initNvonnxparser()
{
#if !TRT_STATIC
    // 动态链接模式
    static LibraryPtr libnvonnxparserPtr{};
    auto fetchPtrs = [](DynamicLibrary* l) {
        // 获取 ONNX Parser 创建函数
        pCreateNvOnnxParserInternal = l->symbolAddress<void*(void*, void*, int)>("createNvOnnxParser_INTERNAL");
        // 获取 ONNX Refitter 创建函数
        pCreateNvOnnxRefitterInternal
            = l->symbolAddress<void*(void*, void*, int)>("createNvOnnxParserRefitter_INTERNAL");
    };
    return initLibrary(libnvonnxparserPtr, kNVONNXPARSER_LIBNAME, fetchPtrs);
#else
    // 静态链接模式
    pCreateNvOnnxParserInternal = createNvOnnxParser_INTERNAL;
    pCreateNvOnnxRefitterInternal = createNvOnnxParserRefitter_INTERNAL;
    return true;
#endif // !TRT_STATIC
}

} // namespace

// ==================== 工厂函数: 创建TensorRT核心对象 ====================

// 创建推理运行时 (IRuntime)
// @return: IRuntime指针,用于反序列化engine和执行推理
//          失败时返回nullptr
IRuntime* createRuntime()
{
    if (!initNvinfer())  // 确保库已加载
    {
        return {};
    }
    ASSERT(pCreateInferRuntimeInternal != nullptr);  // 断言函数指针有效
    // 调用内部API创建Runtime,传入logger和TensorRT版本
    return static_cast<IRuntime*>(pCreateInferRuntimeInternal(&gLogger.getTRTLogger(), NV_TENSORRT_VERSION));
}

// 创建模型构建器 (IBuilder)
// @return: IBuilder指针,用于从ONNX/UFF等模型构建TensorRT Engine
//          失败时返回nullptr
IBuilder* createBuilder()
{
    if (!initNvinfer())
    {
        return {};
    }
    ASSERT(pCreateInferBuilderInternal != nullptr);
    return static_cast<IBuilder*>(pCreateInferBuilderInternal(&gLogger.getTRTLogger(), NV_TENSORRT_VERSION));
}

// 创建模型 refitter (IRefitter)
// @param engine: 要重拟合的已构建engine
// @return: IRefitter指针,用于修改已构建engine的权重
//          失败时返回nullptr
IRefitter* createRefitter(ICudaEngine& engine)
{
    if (!initNvinfer())
    {
        return {};
    }
    ASSERT(pCreateInferRefitterInternal != nullptr);
    // 传入engine对象,logger和版本号
    return static_cast<IRefitter*>(pCreateInferRefitterInternal(&engine, &gLogger.getTRTLogger(), NV_TENSORRT_VERSION));
}

// 创建ONNX解析器 (IParser)
// @param network: 网络定义对象,解析后的模型会填充到这个network中
// @return: ONNX Parser指针,用于解析ONNX模型文件
//          失败时返回nullptr
nvonnxparser::IParser* createONNXParser(INetworkDefinition& network)
{
    if (!initNvonnxparser())
    {
        return {};
    }
    ASSERT(pCreateNvOnnxParserInternal != nullptr);
    // 传入network定义,logger和ONNX Parser版本
    return static_cast<nvonnxparser::IParser*>(
        pCreateNvOnnxParserInternal(&network, &gLogger.getTRTLogger(), NV_ONNX_PARSER_VERSION));
}

// 创建ONNX Refitter (IParserRefitter)
// @param refitter: TensorRT的refitter对象
// @return: ONNX Parser Refitter指针,用于从ONNX模型重拟合engine
//          失败时返回nullptr
nvonnxparser::IParserRefitter* createONNXRefitter(nvinfer1::IRefitter& refitter)
{
    if (!initNvonnxparser())
    {
        return {};
    }
    ASSERT(pCreateNvOnnxRefitterInternal != nullptr);
    return static_cast<nvonnxparser::IParserRefitter*>(
        pCreateNvOnnxRefitterInternal(&refitter, &gLogger.getTRTLogger(), NV_ONNX_PARSER_VERSION));
}

#if ENABLE_UNIFIED_BUILDER

// 处理安全插件库的加载和注册 (统一Builder模式)
// @param safetyPluginRegistry: 安全插件注册表
// @param libPtr: 已加载的动态库指针
// @param pluginArgs: 插件参数 (库名,插件属性列表)
// @return: 成功返回true
bool processSafetyPluginLibrary(nvinfer2::safe::ISafePluginRegistry* safetyPluginRegistry, DynamicLibrary* libPtr,
    samplesSafeCommon::SafetyPluginLibraryArgument const& pluginArgs)
{
    if (libPtr == nullptr)
    {
        sample::gLogError << "Cannot open safety plugin library " << pluginArgs.libraryName << std::endl;
        return false;
    }
    // 插件创建器的符号名称 (固定接口)
    std::string const pluginGetterSymbolName{"getSafetyPluginCreator"};
    // 从动态库中获取插件创建器获取函数
    auto pGetSafetyPluginCreator
        = libPtr->symbolAddress<void*(char const*, char const*)>(pluginGetterSymbolName.c_str());
    if (pGetSafetyPluginCreator == nullptr)
    {
        sample::gLogError << "Cannot find plugin creator getter symbol from plugin library: " << pluginArgs.libraryName
                          << std::endl;
        sample::gLogError << "Please ensure interface function is correctly implemented and exported." << std::endl;
        return false;
    }

    // 遍历插件参数列表,逐个注册插件
    for (auto const& pluginAttr : pluginArgs.pluginAttrs)
    {
        // 调用获取函数,传入namespace和plugin name
        auto pluginCreator = static_cast<IPluginCreatorInterface*>(
            pGetSafetyPluginCreator(pluginAttr.pluginNamespace.c_str(), pluginAttr.pluginName.c_str()));
        if (pluginCreator == nullptr)
        {
            sample::gLogWarning << "Plugin interface getSafetyPluginCreator return nullptr for "
                                << pluginAttr.pluginNamespace << "::" << pluginAttr.pluginName
                                << " in the safety plugin library: " << pluginArgs.libraryName << std::endl;
            sample::gLogWarning
                << "Please ensure interface function is implemented correctly and plugin name/namespace is matched."
                << std::endl;
            continue;
        }
        sample::gLogInfo << "Registering " << pluginAttr.pluginNamespace << "::" << pluginAttr.pluginName
                         << " for TensorRT safety." << std::endl;
        // 注册插件到安全插件注册表
        ErrorCode errorCode
            = safetyPluginRegistry->registerCreator(*pluginCreator, pluginAttr.pluginNamespace.c_str(), *gSafeRecorder);
        if (errorCode != ErrorCode::kSUCCESS)
        {
            sample::gLogWarning << "Failed to register safety plugin " << pluginAttr.pluginNamespace
                                << "::" << pluginAttr.pluginName << std::endl;
            if (errorCode == ErrorCode::kINVALID_ARGUMENT)
            {
                sample::gLogWarning << "Is getPluginName/getPluginNamespace/getPluginVersion interface implemented and "
                                       "return non-nullptr?"
                                    << std::endl;
            }
        }
    }
    return true;
}
#endif // ENABLE_UNIFIED_BUILDER

// ==================== 类型别名 ====================
// 时间点类型 - 用于高精度计时
using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
// 时长类型 - float秒为单位
using duration = std::chrono::duration<float>;

// ==================== 主函数入口 ====================
// trtexec 命令行工具的入口点
// @param argc: 命令行参数个数
// @param argv: 命令行参数数组
// @return: 程序退出码 (EXIT_SUCCESS=0 表示成功)
int main(int argc, char** argv)
{
    // 样本名称 (用于日志和测试框架)
    std::string const sampleName = "TensorRT.trtexec";

    // 定义测试用例 (用于NVIDIA内部测试框架)
    auto sampleTest = sample::gLogger.defineTest(sampleName, argc, argv);

    try
    {
        // 报告测试开始
        sample::gLogger.reportTestStart(sampleTest);

        // ==================== 步骤1: 参数解析 ====================
        // 将命令行参数转换为键值对映射
        Arguments args = argsToArgumentsMap(argc, argv);
        // 创建选项对象 (包含所有可配置选项)
        AllOptions options;

        // 检查是否请求帮助信息 (--help, --helps, -h 等)
        if (parseHelp(args))
        {
            AllOptions::help(std::cout);  // 打印帮助信息
            return EXIT_SUCCESS;          // 直接退出
        }

        // 有命令行参数,进行解析
        if (!args.empty())
        {
            bool failed{false};  // 解析是否失败
            try
            {
                // 解析参数到options对象
                options.parse(args);

                // 检查是否有未识别的参数
                if (!args.empty())
                {
                    AllOptions::help(std::cout);
                    // 打印所有未知选项
                    for (auto const& arg : args)
                    {
                        sample::gLogError << "Unknown option: " << arg.first << " " << arg.second.first << std::endl;
                    }
                    failed = true;
                }
            }
            catch (std::invalid_argument const& arg)
            {
                // 参数解析异常 (如值格式错误)
                AllOptions::help(std::cout);
                sample::gLogError << arg.what() << std::endl;
                failed = true;
            }

            if (failed)
            {
                return sample::gLogger.reportFail(sampleTest);  // 解析失败,返回失败状态
            }
        }
        else
        {
            // 无参数时,默认显示帮助信息
            options.helps = true;
        }

        // 如果请求帮助信息
        if (options.helps)
        {
            AllOptions::help(std::cout);
            return sample::gLogger.reportPass(sampleTest);
        }

        // ==================== 步骤2: 环境设置 ====================
        // 打印选项配置信息
        sample::gLogInfo << options;
        // 设置日志详细程度 (如果请求了verbose)
        if (options.reporting.verbose)
        {
            sample::setReportableSeverity(ILogger::Severity::kVERBOSE);
        }
        // JIT编译版本字符串 (空字符串,用于未来扩展)
        std::string const jitInVersion;
        // 设置CUDA设备 (非CPU模式时)
        if (!options.build.cpuOnly)
        {
            setCudaDevice(options.system.device, sample::gLogInfo);
        }
        // 打印空行分隔
        sample::gLogInfo << std::endl;
        // 打印TensorRT版本信息
        sample::gLogInfo << "TensorRT version: " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "."
                         << NV_TENSORRT_PATCH << jitInVersion << std::endl;

        // 记录指定的运行时模式 (kFULL/kLEANT/kSAFE)
        gUseRuntime = options.build.useRuntime;

        // ==================== 步骤3: 加载插件 ====================
#if !TRT_STATIC
        LibraryPtr nvinferPluginLib{};  // TensorRT标准插件库句柄
#endif /* TRT_STATIC */
        std::vector<LibraryPtr> pluginLibs;  // 用户提供的插件库列表

        // 完整运行时 + 非安全模式: 加载标准插件
        if (gUseRuntime == RuntimeMode::kFULL && !options.build.safe)
        {
            sample::gLogInfo << "Loading standard plugins" << std::endl;
#if !TRT_STATIC
            // 动态加载 libnvinfer_plugin.so
            nvinferPluginLib = loadLibrary(kNVINFER_PLUGIN_LIBNAME);
            // 获取插件初始化函数，通过dlsym获取initLibNvInferPlugins的函数指针
            auto pInitLibNvinferPlugins
                = nvinferPluginLib->symbolAddress<bool(void*, char const*)>("initLibNvInferPlugins");
#else /* TRT_STATIC */
            // 静态链接模式
            auto pInitLibNvinferPlugins = initLibNvInferPlugins;
#endif /* TRT_STATIC */
            ASSERT(pInitLibNvinferPlugins != nullptr);
            // 初始化所有内置插件 (空字符串表示加载全部)
            pInitLibNvinferPlugins(&sample::gLogger.getTRTLogger(), "");

            // 加载用户指定的额外插件库
            for (auto const& pluginPath : options.system.plugins)
            {
                sample::gLogInfo << "Loading supplied plugin library: " << pluginPath << std::endl;
                pluginLibs.emplace_back(loadLibrary(pluginPath));
            }
        }
        // 完整运行时 + 安全模式: 跳过标准插件 (安全模式不支持)
        else if (gUseRuntime == RuntimeMode::kFULL && options.build.safe)
        {
            sample::gLogInfo << "Skipping standard plugin loading due to --safe flag" << std::endl;
        }
        // 其他模式但指定了插件: 报错
        else if (!options.system.plugins.empty())
        {
            throw std::runtime_error("TRT-18412: Plugins require --useRuntime=full.");
        }
        // ==================== 步骤4: 安全插件注册 (统一Builder模式) ====================
#if ENABLE_UNIFIED_BUILDER
        // 获取安全插件注册表
        auto safetyPluginRegistry = sample::safe::getSafePluginRegistry(*gSafeRecorder);
        ASSERT(safetyPluginRegistry != nullptr);

        // 加载用户指定的安全插件库
        if (!options.system.safetyPlugins.empty())
        {
            for (auto const& safetyPluginArg : options.system.safetyPlugins)
            {
                sample::gLogInfo << "Loading supplied safety plugin library with manual registration: "
                                 << safetyPluginArg.libraryName << std::endl;
                auto pluginLib = loadLibrary(safetyPluginArg.libraryName);
                // 处理并注册安全插件
                processSafetyPluginLibrary(safetyPluginRegistry, pluginLib.get(), safetyPluginArg);
                pluginLibs.emplace_back(std::move(pluginLib));
            }
        }
#endif // ENABLE_UNIFIED_BUILDER

        // ==================== 步骤5: 安全模式检查 ====================
        // 检查安全运行时是否可用
        if (options.build.safe && !sample::hasSafeRuntime())
        {
            sample::gLogError << "Safety is not supported because safety runtime library is unavailable." << std::endl;
            return sample::gLogger.reportFail(sampleTest);
        }

        // 一致性检查只适用于安全模式
        if (!options.build.safe && options.build.consistency)
        {
            sample::gLogInfo << "Skipping consistency checker on non-safety mode." << std::endl;
            options.build.consistency = false;
        }

        // 安全模式默认启用强类型
        if (options.build.safe)
        {
            sample::gLogInfo << "StronglyTyped is enabled by default on safety mode." << std::endl;
            options.build.stronglyTyped = true;
        }

        // ==================== 步骤6: CPU-only模式设置 ====================
// Windows does not have setenv call
#if !defined(_WIN32)
        // 设置CPU-only模式环境变量 (仅Linux/Mac)
        if (options.build.cpuOnly)
        {
            // 注意: TRT_INTERNAL_OPTIONS 是TensorRT 10.15的特殊内部选项,未来版本会移除
            sample::gLogInfo << "Setting CPU-only mode" << std::endl;
            char* internalOptions = std::getenv("TRT_INTERNAL_OPTIONS");
            std::string internalOptionsStr;
            if (internalOptions)
            {
                // 追加CPU-only选项
                internalOptionsStr = std::string(internalOptions) + " --cpu_only=1";
            }
            else
            {
                internalOptionsStr = "--cpu_only=1";
            }
            setenv("TRT_INTERNAL_OPTIONS", internalOptionsStr.c_str(), 1);
        }
#endif // !defined(_WIN32)

        // ==================== 步骤7: 构建Engine ====================
        // 创建构建环境对象 (管理构建过程中的临时文件和资源)
        std::unique_ptr<BuildEnvironment> bEnv(new BuildEnvironment(options.build.safe, options.build.versionCompatible,
            options.system.DLACore, options.build.tempdir, options.build.tempfileControls, options.build.leanDLLPath,
            sampleTest.getCmdline()));

        // 实际构建Engine (核心函数)
        // 从模型文件(ONNX/UFF等)构建TensorRT Engine
        bool buildPass = getEngineBuildEnv(options.model, options.build, options.system, *bEnv, sample::gLogError);

        if (!buildPass)
        {
            sample::gLogError << "Engine set up failed" << std::endl;
            return sample::gLogger.reportFail(sampleTest);
        }

#if ENABLE_UNIFIED_BUILDER
        // 设置安全记录器到插件注册表
        safetyPluginRegistry->setSafeRecorder(*gSafeRecorder);
#endif // ENABLE_UNIFIED_BUILDER

        // 如果只请求获取Engine版本号,则在此退出 (版本已在getEngineBuildEnv中打印)
        if (options.build.getPlanVersionOnly)
        {
            return sample::gLogger.reportPass(sampleTest);
        }


        // ==================== 步骤8: Engine反序列化支持检查 ====================
        // dynamicPlugins可能已被getEngineBuildEnv更新
        bEnv->engine.setDynamicPlugins(options.system.dynamicPlugins);
       // 当启用某些选项时,无法在构建平台以外的平台上反序列化Engine
        // 条件: 非安全模式 + 非独立DLA + 运行时平台与构建平台相同
        bool const supportDeserialization = !options.build.safe && !options.build.buildDLAStandalone
            && options.build.runtimePlatform == nvinfer1::RuntimePlatform::kSAME_AS_BUILD;

        // ==================== 步骤9: Engine Refit (权重重拟合) ====================
        // 支持反序列化且Engine可重拟合时
        if (supportDeserialization && options.build.refittable)
        {
            auto* engine = bEnv->engine.get();
            // 显示可重拟合的权重信息
            if (options.reporting.refit)
            {
                dumpRefittable(*engine);
            }
            // 从ONNX模型重拟合Engine
            if (!options.inference.refitOnnxModel.empty())
            {
                bool const success = refitFromOnnx(*engine, options.inference.refitOnnxModel, options.inference.threads);
                if (!success)
                {
                    sample::gLogError << "Engine refit from ONNX model failed." << std::endl;
                    return sample::gLogger.reportFail(sampleTest);
                }
            }
            // 计时重拟合过程
            if (options.inference.timeRefit)
            {
                if (bEnv->network.operator bool())  // 检查network是否可用
                {
                    bool const success = timeRefit(*bEnv->network, *engine, options.inference.threads);
                    if (!success)
                    {
                        sample::gLogError << "Engine refit failed." << std::endl;
                        return sample::gLogger.reportFail(sampleTest);
                    }
                }
                else
                {
                    sample::gLogWarning << "Network not available, skipped timing refit." << std::endl;
                }
            }
        }

        // ==================== 步骤10: 跳过推理检查 ====================
        // 如果用户指定了--skipInference,只构建Engine不运行推理
        if (options.build.skipInference)
        {
            if (supportDeserialization)
            {
                // 打印Layer信息
                printLayerInfo(options.reporting, bEnv->engine.get(), nullptr);
                // 打印优化profile信息
                printOptimizationProfileInfo(options.reporting, bEnv->engine.get());
            }
            sample::gLogInfo << "Skipped inference phase since --skipInference is added." << std::endl;
            return sample::gLogger.reportPass(sampleTest);
        }

        // ==================== 步骤11: 创建推理环境 ====================
        // 推理环境基类指针
        std::unique_ptr<InferenceEnvironmentBase> iEnv;

        // 根据是否安全模式选择不同的推理环境
        if (!options.build.safe)
        {
            // 标准推理环境
            iEnv = std::make_unique<InferenceEnvironmentStd>(*bEnv);
        }
        else
        {
#if ENABLE_UNIFIED_BUILDER
            // 安全推理环境
            iEnv = std::make_unique<InferenceEnvironmentSafe>(*bEnv);
#else
            sample::gLogInfo << "--safe flag is enabled but application is not compatible with safety." << std::endl;
            return sample::gLogger.reportFail(sampleTest);
#endif
        }

        // ==================== 步骤12: 处理动态插件 ====================
        // 避免重复加载已在序列化时包含的动态插件
        // 从所有请求的插件中排除已序列化的插件
        std::vector<std::string> dynamicPluginsNotSerialized;
        for (auto& pluginName : options.system.dynamicPlugins)
        {
            if (std::find(options.system.setPluginsToSerialize.begin(), options.system.setPluginsToSerialize.end(),
                    pluginName)
                == options.system.setPluginsToSerialize.end())
            {
                dynamicPluginsNotSerialized.emplace_back(pluginName);
            }
        }

        // 设置需要加载的动态插件
        iEnv->engine.setDynamicPlugins(dynamicPluginsNotSerialized);
        // 构建环境不再需要,可以释放
        bEnv.reset();

        // ==================== 步骤13: 反序列化计时 (可选) ====================
        // 如果用户请求计时Engine反序列化过程
        if (options.inference.timeDeserialize)
        {
            if (timeDeserialize(*iEnv, options.system))
            {
                return sample::gLogger.reportFail(sampleTest);
            }
            return sample::gLogger.reportPass(sampleTest);
        }

        // ==================== 步骤14: DLA检查 ====================
        // 安全模式 + DLA: 需要保存DLA loadable单独运行
        if (options.build.safe && options.system.DLACore >= 0)
        {
            sample::gLogInfo << "Safe DLA capability is detected. Please save DLA loadable with --saveEngine option, "
                                "then use dla_safety_runtime to run inference with saved DLA loadable, "
                                "or alternatively run with your own application"
                             << std::endl;
            return sample::gLogger.reportFail(sampleTest);
        }

        // ==================== 步骤15: Profiler设置 ====================
        // 检查是否启用了性能分析
        bool const profilerEnabled = options.reporting.profile || !options.reporting.exportProfile.empty();
        // 检查是否启用了Layer信息输出
        bool const layerInfoEnabled = options.reporting.layerInfo || !options.reporting.exportLayerInfo.empty();

        // 安全模式不支持性能分析功能
        if (iEnv->safe && (profilerEnabled || layerInfoEnabled))
        {
            sample::gLogError << "Safe runtime does not support --dumpProfile or --exportProfile=<file> or "
                                 "--dumpLayerInfo or --exportLayerInfo=<file>, please use "
                                 "--verbose to print profiling info."
                              << std::endl;
            return sample::gLogger.reportFail(sampleTest);
        }

        // 配置profiler (如果启用且不是rerun模式)
        if (profilerEnabled && !options.inference.rerun)
        {
            iEnv->profiler.reset(new Profiler);
            // CUDA Graph需要CUDA 11.1+,检查版本兼容性
            if (options.inference.graph && (getCudaDriverVersion() < 11010 || getCudaRuntimeVersion() < 11000))
            {
                options.inference.graph = false;
                sample::gLogWarning
                    << "Graph profiling only works with CUDA 11.1 and beyond. Ignored --useCudaGraph flag "
                       "and disabled CUDA graph."
                    << std::endl;
            }
        }

        // ==================== 步骤16: 设置推理环境 ====================
        // 分配输入缓冲区、设置batch大小等
        if (!setUpInference(*iEnv, options.inference, options.system))
        {
            sample::gLogError << "Inference set up failed" << std::endl;
            return sample::gLogger.reportFail(sampleTest);
        }

        // ==================== 步骤17: 打印Layer和优化信息 ====================
        // 非安全模式打印详细信息
        if (!options.build.safe)
        {
            printLayerInfo(options.reporting, iEnv->engine.get(),
                static_cast<InferenceEnvironmentStd*>(iEnv.get())->contexts.front().get());
            printOptimizationProfileInfo(options.reporting, iEnv->engine.get());
        }

        // ==================== 步骤18: 执行推理 ====================
        std::vector<InferenceTrace> trace;
        sample::gLogInfo << "Starting inference" << std::endl;

        // 实际运行推理
        if (!runInference(options.inference, *iEnv, options.system.device, trace, options.reporting))
        {
            sample::gLogError << "Error occurred during inference" << std::endl;
            return sample::gLogger.reportFail(sampleTest);
        }

        // ==================== 步骤19: 打印性能报告 ====================
        // Profiler启用时,e2e计时不准确 (因为有额外的同步)
        if (profilerEnabled && !options.inference.rerun)
        {
            sample::gLogInfo << "The e2e network timing is not reported since it is inaccurate due to the extra "
                             << "synchronizations when the profiler is enabled." << std::endl;
            sample::gLogInfo
                << "To show e2e network timing report, add --separateProfileRun to profile layer timing in a "
                << "separate run or remove --dumpProfile to disable the profiler." << std::endl;
        }
        else
        {
            // 正常打印性能报告 (吞吐量、延迟等)
            printPerformanceReport(trace, options.reporting, options.inference, sample::gLogInfo, sample::gLogWarning,
                sample::gLogVerbose);
        }

        // 打印推理输出
        printOutput(options.reporting, *iEnv, options.inference.batch);

        // ==================== 步骤20: Profiler rerun (单独运行profiling) ====================
        // 如果启用profiler且请求rerun,单独运行一次用于layer timing
        if (profilerEnabled && options.inference.rerun)
        {
            auto* profiler = new Profiler;
            iEnv->profiler.reset(profiler);
            // 设置profiler到context
            static_cast<InferenceEnvironmentStd*>(iEnv.get())->contexts.front()->setProfiler(profiler);
            static_cast<InferenceEnvironmentStd*>(iEnv.get())->contexts.front()->setEnqueueEmitsProfile(false);
            // 再次检查CUDA版本
            if (options.inference.graph && (getCudaDriverVersion() < 11010 || getCudaRuntimeVersion() < 11000))
            {
                options.inference.graph = false;
                sample::gLogWarning
                    << "Graph profiling only works with CUDA 11.1 and beyond. Ignored --useCudaGraph flag "
                       "and disabled CUDA graph."
                    << std::endl;
            }
            // 再次运行推理 (用于获取layer级别的timing)
            if (!runInference(options.inference, *iEnv, options.system.device, trace, options.reporting))
            {
                sample::gLogError << "Error occurred during inference" << std::endl;
                return sample::gLogger.reportFail(sampleTest);
            }
        }
        // 打印详细性能profile (各layer的耗时)
        printPerformanceProfile(options.reporting, *iEnv);

        // ==================== 结束: 返回成功 ====================
        return sample::gLogger.reportPass(sampleTest);
    }
    // 全局异常捕获
    catch (std::exception const& e)
    {
        sample::gLogError << "Uncaught exception detected: " << e.what() << std::endl;
    }
    // 未知异常或上述失败,返回失败状态
    return sample::gLogger.reportFail(sampleTest);
}

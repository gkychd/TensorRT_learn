#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <cstring>
#include <dlfcn.h>

#include "NvInferRuntime.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"

using namespace nvinfer1;

// Logger for TensorRT messages
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
} gLogger;

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --onnx=<path>       Path to ONNX model (required)\n";
    std::cout << "  --engine=<path>   Path to output TensorRT engine (required)\n";
    std::cout << "  --plugin_lib=<path> Path to custom plugin library (required)\n";
    std::cout << "  --fp16             Use FP16 precision\n";
    std::cout << "  --int8             Use INT8 precision\n";
    std::cout << "  --verbose           Enable verbose logging\n";
    std::cout << std::endl;
}

bool loadPluginLibrary(const std::string& pluginPath) {
    std::cout << "Loading plugin library: " << pluginPath << std::endl;

    void* handle = dlopen(pluginPath.c_str(), RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error loading plugin library: " << dlerror() << std::endl;
        return false;
    }

    std::cout << "Plugin library loaded successfully" << std::endl;
    return true;
}

nvinfer1::ICudaEngine* buildEngine(const std::string& onnxPath, bool fp16, bool int8) {
    // Create builder
    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(gLogger));
    if (!builder) {
        std::cerr << "Failed to create InferBuilder" << std::endl;
        return nullptr;
    }

    // Create network
    uint32_t flags = 1 << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(flags));
    if (!network) {
        std::cerr << "Failed to create Network" << std::endl;
        return nullptr;
    }

    // Create parser (TensorRT 10 uses nvonnxparser::IParser)
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser) {
        std::cerr << "Failed to create ONNX Parser" << std::endl;
        return nullptr;
    }

    // Parse ONNX model from file (TensorRT 10+)
    // Verbosity: 0=INTERNAL_ERROR, 1=ERROR, 2=WARNING, 3=INFO, 4=VERBOSE
    std::cout << "Parsing ONNX model: " << onnxPath << std::endl;
    if (!parser->parseFromFile(onnxPath.c_str(), 0)) {
        std::cerr << "Failed to parse ONNX model" << std::endl;
        // Print parsing errors
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << "Error " << i << ": " << parser->getError(i)->desc() << std::endl;
        }
        return nullptr;
    }

    // Get input tensor name
    int nbInputs = network->getNbInputs();
    std::cout << "Number of inputs: " << nbInputs << std::endl;
    for (int i = 0; i < nbInputs; ++i) {
        auto input = network->getInput(i);
        std::cout << "  Input " << i << ": " << input->getName()
                  << " shape: ";
        auto dims = input->getDimensions();
        for (int j = 0; j < dims.nbDims; ++j) {
            std::cout << dims.d[j] << " ";
        }
        std::cout << std::endl;
    }

    // Get output tensor name
    int nbOutputs = network->getNbOutputs();
    std::cout << "Number of outputs: " << nbOutputs << std::endl;
    for (int i = 0; i < nbOutputs; ++i) {
        auto output = network->getOutput(i);
        std::cout << "  Output " << i << ": " << output->getName() << std::endl;
    }

    // Create config (TensorRT 10+)
    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        std::cerr << "Failed to create BuilderConfig" << std::endl;
        return nullptr;
    }

    // TensorRT 10: setMaxBatchSize is removed, use config
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30);  // 1GB

    // Create optimization profile for dynamic inputs
    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        std::cerr << "Failed to create OptimizationProfile" << std::endl;
        return nullptr;
    }

    // Set optimization profile for dynamic inputs (NCHW format)
    // Input 0: input (N, 3, H, W), Input 1: factor (N, 1, H, W)
    const int batchSize = 1;
    const int height = 224;
    const int width = 224;

    auto input0 = network->getInput(0);
    auto input1 = network->getInput(1);

    profile->setDimensions(input0->getName(), OptProfileSelector::kOPT, Dims4(batchSize, 3, height, width));
    profile->setDimensions(input0->getName(), OptProfileSelector::kMIN, Dims4(1, 3, 1, 1));
    profile->setDimensions(input0->getName(), OptProfileSelector::kMAX, Dims4(8, 3, height, width));

    profile->setDimensions(input1->getName(), OptProfileSelector::kOPT, Dims4(batchSize, 1, height, width));
    profile->setDimensions(input1->getName(), OptProfileSelector::kMIN, Dims4(1, 1, 1, 1));
    profile->setDimensions(input1->getName(), OptProfileSelector::kMAX, Dims4(8, 1, height, width));

    config->addOptimizationProfile(profile);

    std::cout << "Optimization profile set: batch=" << batchSize << ", H=" << height << ", W=" << width << std::endl;

    // Enable FP16 if requested
    if (fp16 && builder->platformHasFastFp16()) {
        std::cout << "Enabling FP16 precision" << std::endl;
        config->setFlag(BuilderFlag::kFP16);
    }

    // Enable INT8 if requested
    if (int8 && builder->platformHasFastInt8()) {
        std::cout << "Enabling INT8 precision" << std::endl;
        config->setFlag(BuilderFlag::kINT8);
    }

    std::cout << "Building TensorRT engine..." << std::endl;

    // TensorRT 10+ API
    auto serializedEngine = builder->buildSerializedNetwork(*network, *config);
    if (!serializedEngine) {
        std::cerr << "Failed to build TensorRT engine" << std::endl;
        return nullptr;
    }

    // Deserialize to get engine
    auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
    auto engine = std::unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size()));

    if (!engine) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return nullptr;
    }

    std::cout << "Engine built successfully!" << std::endl;
    return engine.release();
}

bool saveEngine(nvinfer1::ICudaEngine* engine, const std::string& enginePath) {
    std::cout << "Saving engine to: " << enginePath << std::endl;

    auto serializedEngine = engine->serialize();
    if (!serializedEngine) {
        std::cerr << "Failed to serialize engine" << std::endl;
        return false;
    }

    std::ofstream file(enginePath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for writing: " << enginePath << std::endl;
        return false;
    }

    file.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    file.close();

    std::cout << "Engine saved successfully!" << std::endl;
    return true;
}

int main(int argc, char** argv) {
    std::string onnxPath;
    std::string enginePath;
    std::string pluginLibPath;
    bool fp16 = false;
    bool int8 = false;
    bool verbose = false;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.rfind("--onnx=", 0) == 0) {
            onnxPath = arg.substr(7);
        } else if (arg.rfind("--engine=", 0) == 0) {
            enginePath = arg.substr(9);
        } else if (arg.rfind("--plugin_lib=", 0) == 0) {
            pluginLibPath = arg.substr(14);
        } else if (arg == "--fp16") {
            fp16 = true;
        } else if (arg == "--int8") {
            int8 = true;
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
    }

    // Validate arguments
    if (onnxPath.empty() || enginePath.empty() || pluginLibPath.empty()) {
        std::cerr << "Error: --onnx, --engine, and --plugin_lib are required" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // Load custom plugin library
    // This will automatically register plugins via REGISTER_TENSORRT_PLUGIN macro
    if (!loadPluginLibrary(pluginLibPath)) {
        return 1;
    }

    // Build engine
    auto engine = buildEngine(onnxPath, fp16, int8);
    if (!engine) {
        std::cerr << "Failed to build engine" << std::endl;
        return 1;
    }

    // Save engine
    if (!saveEngine(engine, enginePath)) {
        delete engine;
        return 1;
    }

    delete engine;
    return 0;
}

#include <fstream>
#include <iostream>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <logger.h>

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace sample;

int main(int argc, char** argv)
{
    // Create builder
    Logger m_logger;
    IBuilder* builder = createInferBuilder(m_logger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network (TensorRT 10.x uses IMPLICIT_BATCH by default)
    INetworkDefinition* network = builder->createNetworkV2(0);

    // Parse ONNX file
    IParser* parser = nvonnxparser::createParser(*network, m_logger);
    // 解析onnx到network中
    bool parser_status = parser->parseFromFile("model.onnx", static_cast<int>(ILogger::Severity::kWARNING));

    // Get the name of network input
    Dims dim = network->getInput(0)->getDimensions();
    if (dim.d[0] == -1)  // -1 means it is a dynamic model
    {
        const char* name = network->getInput(0)->getName();
        IOptimizationProfile* profile = builder->createOptimizationProfile();
        profile->setDimensions(name, OptProfileSelector::kMIN, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
        profile->setDimensions(name, OptProfileSelector::kOPT, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
        profile->setDimensions(name, OptProfileSelector::kMAX, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
        config->addOptimizationProfile(profile);
    }

    // Build engine
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 20);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    // Serialize the model to engine file
    IHostMemory* modelStream{ nullptr };
    assert(engine != nullptr);
    modelStream = engine->serialize();

    std::ofstream p("convert_model.engine", std::ios::binary);
    if (!p) {
        std::cerr << "could not open output file to save model" << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    std::cout << "generate file success!" << std::endl;

    // Release resources (TensorRT 10.x uses delete)
    delete modelStream;
    delete engine;
    delete network;
    delete config;
    delete builder;
    return 0;
}

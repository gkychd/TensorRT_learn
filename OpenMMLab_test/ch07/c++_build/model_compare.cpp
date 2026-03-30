#include <fstream>
#include <iostream>
#include <cmath>

#include <NvInfer.h>
#include <logger.h>

#define CHECK(status) \
    do \
    { \
        auto ret = (status); \
        if (ret != 0) \
        { \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort(); \
        } \
    } while (0)

using namespace nvinfer1;
using namespace sample;

static const int IN_H = 224;
static const int IN_W = 224;
static const int BATCH_SIZE = 1;


void doInference(IExecutionContext& context, ICudaEngine& engine, float* input, float* output, int batchSize)
{
    // Get tensor names
    const char* inputName = engine.getIOTensorName(0);
    const char* outputName = engine.getIOTensorName(1);

    // Get tensor dimensions
    Dims inputDims = engine.getTensorShape(inputName);
    Dims outputDims = engine.getTensorShape(outputName);

    // Calculate sizes
    int inputSize = batchSize * inputDims.d[1] * inputDims.d[2] * inputDims.d[3];
    int outputSize = batchSize * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];

    void* buffers[2];

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[0], inputSize * sizeof(float)));
    CHECK(cudaMalloc(&buffers[1], outputSize * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // TensorRT 10.x uses setTensorAddress
    context.setTensorAddress(inputName, buffers[0]);
    context.setTensorAddress(outputName, buffers[1]);

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[0], input, inputSize * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueueV3(stream);
    CHECK(cudaMemcpyAsync(output, buffers[1], outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[0]));
    CHECK(cudaFree(buffers[1]));
}


ICudaEngine* loadEngine(const char* enginePath, IRuntime* runtime)
{
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open engine file: " << enginePath << std::endl;
        return nullptr;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    char* trtModelStream = new char[size];
    file.read(trtModelStream, size);
    file.close();

    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    delete[] trtModelStream;

    return engine;
}


int main(int argc, char** argv)
{
    Logger m_logger;
    IRuntime* runtime = createInferRuntime(m_logger);
    assert(runtime != nullptr);

    // Load both engines
    std::cout << "Loading direct_model.engine..." << std::endl;
    ICudaEngine* engine1 = loadEngine("direct_model.engine", runtime);
    if (!engine1) {
        std::cerr << "Failed to load direct_model.engine" << std::endl;
        return -1;
    }

    std::cout << "Loading convert_model.engine..." << std::endl;
    ICudaEngine* engine2 = loadEngine("convert_model.engine", runtime);
    if (!engine2) {
        std::cerr << "Failed to load convert_model.engine" << std::endl;
        delete engine1;
        delete runtime;
        return -1;
    }

    // Create execution contexts
    IExecutionContext* context1 = engine1->createExecutionContext();
    IExecutionContext* context2 = engine2->createExecutionContext();
    assert(context1 != nullptr);
    assert(context2 != nullptr);

    // Generate input data
    int inputSize = BATCH_SIZE * 3 * IN_H * IN_W;
    float* data = new float[inputSize];
    for (int i = 0; i < inputSize; i++)
        data[i] = static_cast<float>(i % 256) / 255.0f;  // Use varying values

    // Run inference on both engines
    int outputSize = BATCH_SIZE * 3 * IN_H * IN_W / 4;

    float* output1 = new float[outputSize];
    float* output2 = new float[outputSize];

    std::cout << "Running inference on direct_model.engine..." << std::endl;
    doInference(*context1, *engine1, data, output1, BATCH_SIZE);

    std::cout << "Running inference on convert_model.engine..." << std::endl;
    doInference(*context2, *engine2, data, output2, BATCH_SIZE);

    // Compare results
    std::cout << "\nComparing results..." << std::endl;

    float maxDiff = 0.0f;
    float avgDiff = 0.0f;
    int maxDiffIdx = 0;

    for (int i = 0; i < outputSize; i++) {
        float diff = std::abs(output1[i] - output2[i]);
        avgDiff += diff;
        if (diff > maxDiff) {
            maxDiff = diff;
            maxDiffIdx = i;
        }
    }
    avgDiff /= outputSize;

    std::cout << "Output size: " << outputSize << " elements" << std::endl;
    std::cout << "Max difference: " << maxDiff << std::endl;
    std::cout << "Average difference: " << avgDiff << std::endl;
    std::cout << "Max diff index: " << maxDiffIdx << std::endl;

    // Show first few output values
    std::cout << "\nFirst 10 output values:" << std::endl;
    std::cout << "direct:   ";
    for (int i = 0; i < 10; i++)
        std::cout << output1[i] << " ";
    std::cout << std::endl;
    std::cout << "convert:  ";
    for (int i = 0; i < 10; i++)
        std::cout << output2[i] << " ";
    std::cout << std::endl;

    // Determine if results are similar
    float tolerance = 1e-5f;
    if (maxDiff < tolerance) {
        std::cout << "\n[PASS] Results are identical!" << std::endl;
    } else if (maxDiff < 0.01f) {
        std::cout << "\n[WARNING] Results are close but not identical" << std::endl;
    } else {
        std::cout << "\n[FAIL] Results differ significantly!" << std::endl;
    }

    // Cleanup
    delete[] data;
    delete[] output1;
    delete[] output2;
    delete context1;
    delete context2;
    delete engine1;
    delete engine2;
    delete runtime;

    return 0;
}

#include <fstream>
#include <iostream>

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
    // TensorRT 10.x uses getNbIOTensors instead of getNbBindings
    assert(engine.getNbIOTensors() == 2);

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

int main(int argc, char** argv)
{
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{ nullptr };
    size_t size{ 0 };

    std::ifstream file("direct_model.engine", std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    Logger m_logger;
    IRuntime* runtime = createInferRuntime(m_logger);
    assert(runtime != nullptr);
    // TensorRT 10.x deserializeCudaEngine only takes 2 arguments
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // generate input data
    int inputSize = BATCH_SIZE * 3 * IN_H * IN_W;
    float* data = new float[inputSize];
    for (int i = 0; i < inputSize; i++)
        data[i] = 1;

    // Run inference
    int outputSize = BATCH_SIZE * 3 * IN_H * IN_W / 4;
    float* prob = new float[outputSize];
    doInference(*context, *engine, data, prob, BATCH_SIZE);

    std::cout << "Inference done!" << std::endl;

    // Destroy the engine (TensorRT 10.x uses delete)
    delete context;
    delete engine;
    delete runtime;
    delete[] trtModelStream;
    delete[] data;
    delete[] prob;
    return 0;
}

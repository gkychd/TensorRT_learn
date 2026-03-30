cmake_minimum_required(VERSION 3.16)
project(direct_build)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TensorRT path in docker
set(TENSORRT_ROOT "/TensorRT-10.15.1.29")

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(${TENSORRT_ROOT}/include)
include_directories(${TENSORRT_ROOT}/../workspace/TensorRT/samples/common)
include_directories(${CUDA_INCLUDE_DIRS})

# Link directories
link_directories(${TENSORRT_ROOT}/lib)
link_directories(/usr/local/cuda-12.6/targets/x86_64-linux/lib)

# Link libraries
set(LIBRARIES
    nvinfer
    nvonnxparser
    ${CUDA_LIBRARIES}
    cudart
    pthread
    dl
)

# Executable: direct_build
add_executable(direct_build direct_build.cpp)
target_link_libraries(direct_build ${LIBRARIES})

# Executable: convert_build
add_executable(convert_build convert_build.cpp)
target_link_libraries(convert_build ${LIBRARIES})

# Executable: model_infer
add_executable(model_infer model_infer.cpp)
target_link_libraries(model_infer ${LIBRARIES})

# Executable: model_compare
add_executable(model_compare model_compare.cpp)
target_link_libraries(model_compare ${LIBRARIES})

# Copy model.onnx to build folder
file(COPY ${CMAKE_SOURCE_DIR}/model.onnx DESTINATION ${CMAKE_BINARY_DIR})

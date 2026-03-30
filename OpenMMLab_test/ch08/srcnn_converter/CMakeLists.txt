# CMakeLists.txt for SRCNN ONNX to TensorRT Converter

cmake_minimum_required(VERSION 3.16)
project(srcnn_converter LANGUAGES CXX CUDA)

if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
endif()

# Define set_ifndef macro before using
macro(set_ifndef var val)
    if(NOT ${var})
        set(${var} ${val})
    endif()
endmacro()

# -------- CONFIGURATION --------
if(NOT MSVC)
    set_ifndef(TRT_LIB /usr/lib/x86_64-linux-gnu)
    set_ifndef(TRT_INCLUDE /usr/include/x86_64-linux-gnu)
    set_ifndef(CUDA_INC_DIR /usr/local/cuda/include)
    set_ifndef(CUDA_LIB_DIR /usr/local/cuda)
endif()

# Find TensorRT
find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})

find_library(_NVONNX_PARSER_LIB nvonnxparser HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64)
set_ifndef(NVONNX_PARSER_LIB ${_NVONNX_PARSER_LIB})

find_library(_CUDART_LIB cudart HINTS ${CUDA_LIB_DIR} PATH_SUFFIXES lib lib64)
set_ifndef(CUDART_LIB ${_CUDART_LIB})

# Include directories
include_directories(
    ${CUDA_INC_DIR}
    ${TRT_INCLUDE}
)

# Find ONNX model
set(ONNX_MODEL "${CMAKE_SOURCE_DIR}/../srcnn3.onnx")

# Get absolute path to plugin library (use SOVERSION file directly)
set(PLUGIN_LIB_SOURCE "/${CMAKE_SOURCE_DIR}/../backend_ops/tensorrt/build/libmmdeploy_tensorrt_ops.so")
get_filename_component(PLUGIN_LIB_ABS "${PLUGIN_LIB_SOURCE}" ABSOLUTE)

set(OUTPUT_ENGINE "${CMAKE_BINARY_DIR}/srcnn3.engine")

# Source files
add_executable(srcnn_converter srcnn_converter.cpp)

# Link libraries
target_link_libraries(srcnn_converter PRIVATE ${NVINFER_LIB})
target_link_libraries(srcnn_converter PRIVATE ${NVONNX_PARSER_LIB})
target_link_libraries(srcnn_converter PRIVATE ${CUDART_LIB})
target_link_libraries(srcnn_converter PRIVATE ${NVINFER_LIB})  # for initLibNvInferPlugins

# Custom plugin library
# add_custom_command(
#     OUTPUT ${OUTPUT_ENGINE}
#     COMMAND ${CMAKE_CURRENT_BINARY_DIR}/srcnn_converter
#             --onnx=${ONNX_MODEL}
#             --engine=${OUTPUT_ENGINE}
#             --plugin_lib=${PLUGIN_LIB_ABS}
#     DEPENDS srcnn_converter
#     COMMENT "Building TensorRT engine from ONNX model..."
#     VERBATIM
# )

# add_custom_target(engine ALL DEPENDS ${OUTPUT_ENGINE})

message(STATUS "ONNX Model: ${ONNX_MODEL}")
message(STATUS "Output Engine: ${OUTPUT_ENGINE}")

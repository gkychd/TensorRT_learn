# Copyright (c) OpenMMLab. All rights reserved.
# TensorRT Custom Operators - Standalone Build

cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(mmdeploy_tensorrt_ops LANGUAGES CXX CUDA)

if(NOT MSVC) #MSVC是Windows下的编译
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wno-deprecated-declarations")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")
endif()

# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if(NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# -------- CONFIGURATION --------
if(NOT MSVC)
    set_ifndef(TRT_LIB /usr/lib/x86_64-linux-gnu)
    set_ifndef(TRT_INCLUDE /usr/include/x86_64-linux-gnu)
    set_ifndef(CUDA_INC_DIR /usr/local/cuda/include)
    set_ifndef(CUDA_LIB_DIR /usr/local/cuda)
    set_ifndef(CUBLAS_LIB_SUFFIXES "lib;lib64")
    set_ifndef(CUDNN_DIR /usr) #当前docker环境下没有cudnn 暂不考虑
endif()

# Find dependencies
message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")

# TensorRT's nvinfer lib
# 查找名为 nvinfer 的库文件
# ${TRT_LIB} 是搜索路径提示（TensorRT 库目录）
# PATH_SUFFIXES lib lib64 是额外的搜索子目录
# 结果存到内部变量 _NVINFER_LIB
find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64)
# 如果变量 NVINFER_LIB 未定义，则将其设置为找到的库路径
# 这是一种保护性赋值：用户可以提前指定 NVINFER_LIB，否则自动使用找到的路径
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})

find_library(_CUDART_LIB cudart HINTS ${CUDA_LIB_DIR} PATH_SUFFIXES lib lib64)
set_ifndef(CUDART_LIB ${_CUDART_LIB})

find_library(_CUBLAS_LIB cublas HINTS ${CUDA_LIB_DIR} PATH_SUFFIXES ${CUBLAS_LIB_SUFFIXES})
set_ifndef(CUBLAS_LIB ${_CUBLAS_LIB})

find_library(_CUDA_LIB cuda HINTS ${CUDA_LIB_DIR} PATH_SUFFIXES lib/stubs lib64/stubs)
set_ifndef(CUDA_LIB ${_CUDA_LIB})

find_library(_CUDNN_LIB cudnn HINTS ${CUDNN_DIR} PATH_SUFFIXES lib lib64)
if(_CUDNN_LIB)
    set(CUDNN_LIB ${_CUDNN_LIB})
    message(STATUS "CUDNN_LIB: ${CUDNN_LIB}")
else()
    message(STATUS "cuDNN not found, some plugins will be skipped")
endif()

# -------- BUILDING --------

add_definitions(-DTENSORRT_BUILD_LIB)

# Add include directories
file(REAL_PATH ${CMAKE_SOURCE_DIR} CMAKE_SOURCE_DIR_REALPATH)

set(BACKEND_OPS_DIR ${CMAKE_SOURCE_DIR_REALPATH})
set(COMMON_DIR ${CMAKE_SOURCE_DIR_REALPATH}/common)
set(COMMON_IMPL_DIR ${CMAKE_SOURCE_DIR_REALPATH}/common_impl)

include_directories(
    ${CUDA_INC_DIR}
    ${TRT_INCLUDE}
    ${BACKEND_OPS_DIR}
    ${COMMON_DIR}
    ${COMMON_IMPL_DIR}
)

if(CUDNN_DIR AND EXISTS ${CUDNN_DIR}/include)
    include_directories(${CUDNN_DIR}/include)
endif()

# Define list of operators to build
# Note: Some operators require cuDNN or external dependencies
set(BACKEND_OPS
    batched_nms
    batched_rotated_nms
    bicubic_interpolate
    gather_topk
    grid_priors
    grid_sampler
    multi_level_roi_align
    multi_level_rotated_roi_align
    multi_scale_deform_attn
    roi_align
    scatternd
    dynamic_resize
)

# Add cuDNN-dependent operators if cuDNN is available
if(CUDNN_LIB)
    list(APPEND BACKEND_OPS
        instance_norm
        scaled_dot_product_attention
        modulated_deform_conv
        deform_conv
    )
    message(STATUS "cuDNN found, building cuDNN-dependent operators")
endif()

# Build list of source files
set(ALL_SRCS)

# Add common_impl sources
file(GLOB_RECURSE COMMON_IMPL_SRCS
    ${COMMON_IMPL_DIR}/*.cpp
    ${COMMON_IMPL_DIR}/*.cu
)
list(APPEND ALL_SRCS ${COMMON_IMPL_SRCS})

# Add operator sources
foreach(OP ${BACKEND_OPS})
    if(EXISTS ${BACKEND_OPS_DIR}/${OP})
        file(GLOB OP_SRCS
            ${BACKEND_OPS_DIR}/${OP}/*.cpp
            ${BACKEND_OPS_DIR}/${OP}/*.cu
        )
        list(APPEND ALL_SRCS ${OP_SRCS})
        message(STATUS "Adding operator: ${OP}")
    else()
        message(WARNING "Operator directory not found: ${OP}")
    endif()
endforeach()

message(STATUS "Building operators: ${BACKEND_OPS}")

# Define plugin library target
add_library(${PROJECT_NAME} SHARED ${ALL_SRCS})

# Use C++17
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

# Enable relaxed constexpr for CUDA
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${NVINFER_LIB})
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDART_LIB})
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUBLAS_LIB})
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIB})
if(CUDNN_LIB)
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DUSE_CUDNN)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDNN_LIB})
endif()

# Set output name
set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME "mmdeploy_tensorrt_ops"
    VERSION ${TensorRT_VERSION}
    SOVERSION ${TRT_MAJOR}
)

message(STATUS "NVINFER_LIB: ${NVINFER_LIB}")

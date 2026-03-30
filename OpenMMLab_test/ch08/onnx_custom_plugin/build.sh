#!/bin/bash
rm -rf build
mkdir build && cd build
cmake .. -DTRT_INCLUDE=/workspace/TensorRT/include -DTRT_LIB=/usr/lib64
make -j
#!/bin/bash
cd /workspace/TensorRT/OpenMMLab_test/ch09_simple

# 使用 --exportOutput 导出输出到 JSON 文件
/workspace/TensorRT/build/out/trtexec --loadEngine=attention_engine.trt \
    --loadInputs=query:attention_inputs/query.bin,key:attention_inputs/key.bin,value:attention_inputs/value.bin,mask:attention_inputs/mask.bin \
    --exportOutput=output.json

import onnx  
 
onnx.utils.extract_model('whole_model.onnx', 'partial_model.onnx', ['22'], ['28']) 

#添加额外输出
onnx.utils.extract_model('whole_model.onnx', 'submodel_1.onnx', ['22'], ['27', '31']) 

#添加冗余输入
onnx.utils.extract_model('whole_model.onnx', 'submodel_2.onnx', ['22', 'input.1'], ['28']) 

#输入信息不足，会报错，这里是因为24和26作为共同输入给到27，而这里只有24，因此会报错 
#onnx.utils.extract_model('whole_model.onnx', 'submodel_3.onnx', ['24'], ['28']) 
onnx.utils.extract_model('whole_model.onnx', 'submodel_3.onnx', ['24', '26'], ['28']) 

#输出onnx中间节点的值
onnx.utils.extract_model('whole_model.onnx', 'more_output_model.onnx', ['input.1'], ['31', '23', '25', '27'])
#提取不同段的子模型 以下4个模型组合在一起，等价于['input.1'], ['31']
onnx.utils.extract_model('whole_model.onnx', 'debug_model_1.onnx', ['input.1'], ['23']) 
onnx.utils.extract_model('whole_model.onnx', 'debug_model_2.onnx', ['23'], ['25']) 
onnx.utils.extract_model('whole_model.onnx', 'debug_model_3.onnx', ['23'], ['27']) 
onnx.utils.extract_model('whole_model.onnx', 'debug_model_4.onnx', ['25', '27'], ['31']) 


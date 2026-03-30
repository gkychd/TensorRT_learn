import onnxruntime 
import torch 
import numpy as np 
 
class Model(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
 
    def forward(self, x): 
        return torch.asinh(x) 
 
model = Model() 
input = torch.rand(1, 3, 10, 10) 
#- `model(input)`: 执行 [model] 的前向传播计算，输入数据为 [input]。
#- `.detach()`: 将生成的张量从计算图中分离，阻止梯度回溯，因为此处仅需推理结果。
#- `.numpy()`: 将 PyTorch Tensor 转换为 NumPy 数组，以兼容 `onnxruntime` 的输出格式。
torch_output = model(input).detach().numpy() 
 
sess = onnxruntime.InferenceSession('asinh.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']) 
ort_output = sess.run(None, {'0': input.numpy()})[0] 
 
assert np.allclose(torch_output, ort_output) #torch结果与onnx结果一致
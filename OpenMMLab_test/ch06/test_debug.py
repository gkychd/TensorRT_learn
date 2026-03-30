from debug_tool import DebugOp, debug_apply, Debugger
from types import MethodType 
import torch

class Model(torch.nn.Module): 
 
    def __init__(self): 
        super().__init__() 
        self.convs1 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, 1, 1), 
                                          torch.nn.Conv2d(3, 3, 3, 1, 1), 
                                          torch.nn.Conv2d(3, 3, 3, 1, 1)) 
        self.convs2 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, 1, 1), 
                                          torch.nn.Conv2d(3, 3, 3, 1, 1)) 
        self.convs3 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, 1, 1), 
                                          torch.nn.Conv2d(3, 3, 3, 1, 1)) 
        self.convs4 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, 1, 1), 
                                          torch.nn.Conv2d(3, 3, 3, 1, 1), 
                                          torch.nn.Conv2d(3, 3, 3, 1, 1)) 
 
    def forward(self, x): 
        x = self.convs1(x) 
        x = self.convs2(x) 
        x = self.convs3(x) 
        x = self.convs4(x) 
        return x 
 
torch_model = Model() 

#以上为原始模式
#以下为为原始模型增加debug节点
debugger = Debugger() 
def new_forward(self, x): 
    x = self.convs1(x) 
    x = debugger.debug(x, 'x_0') 
    x = self.convs2(x) 
    x = debugger.debug(x, 'x_1') 
    x = self.convs3(x) 
    x = debugger.debug(x, 'x_2') 
    x = self.convs4(x) 
    x = debugger.debug(x, 'x_3') 
    return x 
 
torch_model.forward = MethodType(new_forward, torch_model)

dummy_input = torch.randn(1, 3, 10, 10) 
#在调用torch.onnx.export会执行一次forward，从而执行到debug，在该接口里会保存需要debug的层的torch结果
torch.onnx.export(torch_model, dummy_input, 'before_debug.onnx', input_names=['input']) 

#before_debug.onnx中用my::Debug算子是自定义节点，在onnx中无法解析
#通过extract_debug_model将debug节点转化为Identity onnx标准算子，生成after_debug.onnx
debugger.extract_debug_model('before_debug.onnx', 'after_debug.onnx') 
debugger.run_debug_model({'input':dummy_input.numpy()}, 'after_debug.onnx')
debugger.print_debug_result() 
import torch 
import onnx 
 
onnx_model = 'model.onnx' 

class NaiveModel(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.pool = torch.nn.MaxPool2d(2, 2) 
 
    def forward(self, x): 
        return self.pool(x) 
 
device = torch.device('cuda:0') 
 
# generate ONNX model 
torch.onnx.export(NaiveModel(), torch.randn(1, 3, 224, 224), onnx_model, input_names=['input'], output_names=['output'], opset_version=11) 


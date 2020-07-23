import torch
import torchvision
from torch2trt import torch2trt
import os

class Quantizer:
    def __init__(self, quant_level):
        self.quant_level = quant_level
    
    def quantized_model(self, model):
        os.chdir("/home/goel39/torch2trt")
        if model == "VGG16":
            net = torchvision.models.VGG('VGG16', pretrained = True)
        if model == "VGG19":
            net = torchvision.models.VGG('VGG19', pretrained = True)
        if model == "RESNET":
            net = torchvision.models.ResNet18(pretrained = True)
        if model == "MOBILENET":
            net = torchvision.models.MobileNetV2(pretrained = True)
        if model == "EFFICIENTNET":
            net = torchvision.models.EfficientNetB0(pretrained = True)
        
        if quant_level == "float16":
            model_trt = torch2trt(model, [x], fp16_mode=True)
        
        if quant_level == "int8":
            model_trt = torch2trt(model, [x], int8_mode=True)

        return model_trt
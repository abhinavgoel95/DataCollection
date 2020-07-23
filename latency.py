import cv2
import torch
import time
import math
#from quantizer import Quantizer

class Latency:
    def __init__(self, video, net, fps, samp_rate, quant, res, length, model_name, device):
        self.cap = video
        self.net = net
        self.quant = quant  
        self.fps = fps
        self.res = res
        self.samp_rate = samp_rate
        self.length = length
        self.device = device
        self.model_name = model_name
        
    def measure_latency(self):
        if self.quant == "float32":
            return self._measure_unquant()
        else:
            return self._measure_quant()
        
    def _measure_unquant(self):    
        start = time.time()
        count = 0
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                if count <= (self.samp_rate)*self.length*self.fps:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = torch.from_numpy(frame).clone().detach().float().to(self.device).reshape(1, 3, int(math.sqrt(frame.size/3)), int(math.sqrt(frame.size/3)))
                    self.net(frame)
                    count+=1
                else:
                    break
            else: 
                break
        time_taken = time.time() - start
        return count, time_taken
    
    def _measure_quant(self):
        quant = Quantizer(self.quant)
        net = quant.quantized_model(self.model)
        start = time.time()
        count = 0
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                if count <= (self.samp_rate)*self.length*self.fps:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = torch.from_numpy(frame).clone().detach().float().to(self.device).reshape(1, 3, int(math.sqrt(frame.size/3)), int(math.sqrt(frame.size/3)))
                    self.net(frame)
                    count+=1
                else:
                    break
            else: 
                break
        time_taken = time.time() - start
        os.chdir("/home/goel39/Qualcomm/data_collection/imagenet")
        return count, time_taken
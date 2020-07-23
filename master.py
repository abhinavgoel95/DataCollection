import os
import numpy as np
import csv
import os.path
from os import path
import sys

models = ["VGG16", "VGG19", "RESNET", "MOBILENET", "EFFICIENTNET"]
pruning = list(np.arange(0,1.1,0.1))
quant = ['float32', 'float16', 'int8']
res = list(np.arange(0.3, 5,0.2))
sampling_rate = list(np.arange(0.1, 1.1, 0.1))
cpu_only = [False, True]

filename = "csv_files/{network_name}_data.csv".format(network_name=models[0])

fields = ['model', 'prune', 'quant', 'test_batch_size', 'sampling_rate', 'input_resolution', 'accuracy', 'output_fps', 'macs'] 
with open(filename, 'w') as csvfile:  
    csvwriter = csv.writer(csvfile)  
    csvwriter.writerow(fields)

for i in pruning:
    for j in res:
        for k in sampling_rate:
            os.system("python main.py --resume saved_models/model_{model_name}_{quant}.pth.tar --benchmarks --prune {pruning} --resolution {resolution} --sampling_rate {samp} --output_file {fname}".format(model_name=models[0], pruning=i, resolution=j, samp=k, fname=filename, quant = "float32"))
            
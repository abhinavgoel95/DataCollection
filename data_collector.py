import os
import numpy as np
import csv
import os.path
from os import path
import sys

for index in [0]:
    print(index)
    models = ["SHUFFLENET", "VGG16", "VGG19", "RESNET", "MOBILENET", "INCEPTION", "DENSENET"]
    pruning = list(np.arange(0., 1.1, 0.1))
    quant = ['float32', 'float16', 'int8']
    res = list(np.arange(1, 5.1, 0.5))
    #sampling_rate = list(np.arange(0.5/60, 1.1, 0.5/60))
    sampling_rate = list(np.arange(0.1, 1.1, 0.1))

    filename = "csv_files/{network_name}_data.csv".format(network_name=models[index])

    fields = ['model', 'prune', 'quant', 'test_batch_size', 'sampling_rate', 'input_resolution', 'accuracy', 'output_fps', 'macs'] 
    with open(filename, 'w') as csvfile:  
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(fields)

    for i in res:
        for j in pruning:
            for k in sampling_rate:
                os.system("python main.py --benchmarks --resolution {res} --fps 30 --prune {prune} --sampling_rate {samp} --output_file {fname} --model {model_name}".format(samp = k, model_name=models[index], res=i, prune = j, fname=filename, quant = "float32"))
#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from interpolation import three_nn, three_nn_gpu

inp1 = np.random.random((3, 2048, 3))
inp2 = np.random.random((3, 512, 3))

print("Running CPU...")
cpu = three_nn(inp1, inp2)
print("Running GPU...")
gpu = three_nn_gpu(inp1, inp2)
gpu = np.array(gpu)
cpu = np.array(cpu)

if np.isclose(cpu, gpu).all():
    print("CPU and GPU gave equal (within tolerance) results!")
else:
    print("CPU and GPU disagree on results")

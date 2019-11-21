#!/usr/bin/env python3
import sys

try:
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_dir = sys.argv[3]
except:
    print("Usage: {} <input_dir> <output_dir> <model_dir>".format(sys.argv[0]))
    exit(-1)

import model
import h5py
import numpy as np
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sys.path.append('./io'); 
from io_utils import *
from settings import *
import tensorflow as tf 

model = tf.keras.models.load_model(model_dir)

test_dirs = open_file_list(input_dir, "test_files.txt")

for input_file in test_dirs:
    output_file = os.path.join(output_dir, "{}_output.h5".format(os.path.splitext(os.path.basename(input_file))[0]))
    print("{}:".format(output_file))
    with h5py.File(input_file, 'r') as input_file, h5py.File(output_file, 'w') as output_file:
        output_file.create_dataset("data", data=input_file["data"])
        output_file.create_dataset("label", data=input_file["label"])
        predictions = model.predict(input_file["data"], verbose=1)
        predictions_sparse = np.argmax(predictions, axis=2)
        output_file.create_dataset("prediction", data=predictions_sparse)

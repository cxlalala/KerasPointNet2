import sys

try:
    top_dir = sys.argv[1]
    bottom_dir = sys.argv[2]
    left_dir = sys.argv[3]
    right_dir = sys.argv[4]
    output_dir = sys.argv[5]
    model_dir = sys.argv[6]
except:
    print("Usage: {} <top> <bottom> <left> <right> <output_dir> <model_dir>".format(sys.argv[0]))
    exit(-1)

import model
import h5py
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from io_utils.depthmap_dataset import yield_sampled_chunks
import tensorflow as tf 

from datetime import datetime

print("Loading model...")
model = tf.keras.models.load_model(model_dir)

output_file = os.path.join(output_dir, "{}_output.h5".format("13296"))

def gen():
    return yield_sampled_chunks(top_dir, bottom_dir, left_dir, right_dir, 20, 2048, 18000.0)

print("Evaluating...")
with h5py.File(output_file, 'w') as output_file:
    data = np.array(list(gen()))
    predictions = model.predict(data)
    predictions_sparse = np.argmax(predictions, axis=2)
    output_file.create_dataset("data", data=data)
    output_file.create_dataset("prediction", data=predictions_sparse)

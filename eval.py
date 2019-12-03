#!/usr/bin/env python3
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

point_dataset = tf.data.Dataset.from_generator(gen, np.float32, (2048, 3))
point_dataset = point_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
point_dataset = point_dataset.map(lambda data: tf.expand_dims(data, axis=0))

with h5py.File(output_file, 'w') as output_file:
    predictions = model.predict_generator(point_dataset, verbose=1)
    predictions_sparse = np.argmax(predictions, axis=2)
    data = np.array(list(gen()))
    output_file.create_dataset("data", data=data)
    output_file.create_dataset("prediction", data=predictions_sparse)

#for batch in chunks:
#    print(batch.shape)
#    batch = np.expand_dims(batch, axis=0)
#    print(batch.shape)
#    predictions = model.predict_on_batch(batch)

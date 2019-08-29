#!/usr/bin/env python2
import tensorflow as tf
import sys
import os
import h5py
from tensorflow import keras
import numpy as np
sys.path.append('./io'); from h5_dataset import dataset_from_h5_files
sys.path.append('./layers'); from layers import *

# Act like V2.0
tf.enable_eager_execution()

def get_model(n_points, n_channels, n_classes):
    input_layer = keras.layers.Input(shape=(n_points, n_channels))
    l0_xyz = input_layer
    l0_points = None
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1, nsample=32, mlp=[32, 32, 64])
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, mlp=[128, 128, 128])

    net = tf.keras.layers.Conv1D(128, 1, activation='relu')(l0_points)
    #net = tf.keras.layers.Dropout(0.5)(net)
    net = tf.keras.layers.Conv1D(n_classes, 1, activation=None)(net)

    return keras.models.Model(inputs=input_layer, outputs=net)

def open_file_list(directory, name):
    full_file_path = os.path.join(directory, name)
    try:
        training_list = file(full_file_path, "r")
        return [os.path.join(directory, line[:-1]) for line in training_list]
    except:
        print "Expected a training list at {}".format(full_file_path)
        exit(-1)

try:
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    print(output_dir)
except:
    print "Usage: {} <dir> <output dir>".format(sys.argv[0])

train_dirs = open_file_list(input_dir, "train_files.txt")
test_dirs = open_file_list(input_dir, "test_files.txt")

dataset = dataset_from_h5_files(train_dirs)

model = get_model(1024, 3, 2)

def sparse_loss(target, output):
    return keras.backend.sparse_categorical_crossentropy(target, output, from_logits=True)

optimizer=tf.keras.optimizers.Adam(0.001)
model.compile(optimizer=optimizer,
              loss=sparse_loss,
              metrics=['accuracy'])

#keras.utils.plot_model(model, to_file='model.png')
model.fit_generator(dataset.batch(1).shuffle(5000))

for input_file in test_dirs:
    output_file = os.path.join(output_dir, "{}_output.h5".format(os.path.splitext(os.path.basename(input_file))[0]))
    print(output_file)
    with h5py.File(input_file, 'r') as input_file, h5py.File(output_file, 'w') as output_file:
        output_file.create_dataset("data", data=input_file["data"])
        predictions = (np.argmax(model.predict(input_file["data"]), axis=2) * 2) + input_file["label"]
        output_file.create_dataset("label", data=predictions)

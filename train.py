#!/usr/bin/env python2
import tensorflow as tf
import sys
import os
import h5py
from tensorflow import keras
import numpy as np
sys.path.append('./io'); from h5_dataset import H5FilesDatasetGenerator
sys.path.append('./layers'); from layers import *

# Act like V2.0
tf.enable_eager_execution()

def get_model(n_points, n_channels, n_classes):
    input_layer = keras.layers.Input(shape=(n_points, n_channels))
    l0_xyz = input_layer
    l0_points = None
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1, nsample=32, mlp=[32, 32, 64])
    #l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=256, radius=0.2, nsample=32, mlp=[64, 64, 128])
    #l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=64, radius=0.4, nsample=32, mlp=[128, 128, 256])
    #l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=16, radius=0.8, nsample=32, mlp=[256, 256, 512])
    #l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256, 256])
    #l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256, 256])
    #l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256, 128])
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, mlp=[128, 128, 128])

    net = tf.keras.layers.Conv1D(128, 1, activation='relu')(l0_points)
    net = tf.keras.layers.Dropout(0.5)(net)
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
except:
    print "Usage: {} <dir> <output dir>".format(sys.argv[0])

train_dirs = open_file_list(input_dir, "train_files.txt")
test_dirs = open_file_list(input_dir, "test_files.txt")

train_generator = H5FilesDatasetGenerator(train_dirs, [("data", None), ("label", lambda batch: np.expand_dims(batch, axis=2))])
train_dataset = tf.data.Dataset.from_generator(train_generator, train_generator.dtypes, train_generator.shapes)

model = get_model(1024, 3, 2)

def sparse_loss(target, output):
    return keras.backend.sparse_categorical_crossentropy(target, output, from_logits=True)
    #return keras.losses.sparse_categorical_crossentropy(target, output)

optimizer=tf.keras.optimizers.Adam(0.1)

#keras.utils.plot_model(model, to_file='model.png')

ckpt = tf.train.Checkpoint(optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)

if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

model.compile(optimizer=optimizer,
              #loss='sparse_categorical_crossentropy',
              loss=sparse_loss,
              metrics=['sparse_categorical_accuracy'])

#model.summary(line_length=212)

for _ in range(20):
    model.fit_generator(train_dataset.batch(1).shuffle(8000), steps_per_epoch=train_generator.total_samples)
    
    save_path = manager.save()
    print("Save path: {0}".format(save_path))
    
    for input_file in test_dirs:
        output_file = os.path.join(output_dir, "{}_output.h5".format(os.path.splitext(os.path.basename(input_file))[0]))
        print(output_file)
        with h5py.File(input_file, 'r') as input_file, h5py.File(output_file, 'w') as output_file:
            output_file.create_dataset("data", data=input_file["data"])
            predictions = model.predict(input_file["data"])
            predictions_sparse = (np.argmax(predictions, axis=2) * 2) + input_file["label"]
            output_file.create_dataset("label", data=predictions_sparse)

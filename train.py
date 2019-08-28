#!/usr/bin/env python2
import tensorflow as tf
import sys
from tensorflow import keras
sys.path.append('./io'); from h5_dataset import dataset_from_h5_files
sys.path.append('./layers'); from layers import *

# Act like V2.0
tf.enable_eager_execution()

def get_model(n_points, n_channels, n_classes):
    input_layer = keras.layers.Input(shape=(n_points, n_channels))
    l0_xyz = input_layer
    l0_points = None
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=n_points, radius=0.1, nsample=32, mlp=[32,32,64])
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128])

    net = tf.keras.layers.Conv1D(128, 1, activation='relu')(l0_points)
    net = tf.keras.layers.Dropout(0.5)(net)
    net = tf.keras.layers.Conv1D(n_classes, 1, activation=None)(net)

    return keras.models.Model(inputs=input_layer, outputs=net)

dataset = dataset_from_h5_files(sys.argv[1:])
model = get_model(1024, 3, 2)

def sparse_loss(target, output):
    return keras.backend.sparse_categorical_crossentropy(target, output, from_logits=False)

optimizer=tf.keras.optimizers.Adam(0.001)
model.compile(optimizer=optimizer,
              loss=sparse_loss,
              metrics=['accuracy'])

model.fit_generator(dataset.batch(1).shuffle(5000))
#model.fit_generator(dataset)

#!/usr/bin/env python2
import sys
try:
    input_dir = sys.argv[1]
except:
    print "Usage: {} <dataset dir> <checkpoint_dir>".format(sys.argv[0])
    exit(-1)

try:
    checkpoint_dir = sys.argv[2]
except:
    checkpoint_dir = 'tf_ckpts/best.ckpt'

import tensorflow as tf
import os
import h5py
import numpy as np
import model
sys.path.append('./io'); 
from h5_dataset import H5FilesDatasetGenerator
from io_utils import *

# Act like V2.0
tf.enable_eager_execution()

# Start dataset generators 
train_dirs = open_file_list(input_dir, "train_files.txt")
test_dirs = open_file_list(input_dir, "test_files.txt")

def dataset_from_h5_files(filenames):
    data_generator = H5FilesDatasetGenerator(filenames, "data")
    label_generator = H5FilesDatasetGenerator(filenames, "label")

    data_dataset = tf.data.Dataset.from_generator(data_generator, data_generator.dtype, data_generator.shape)
    label_dataset = tf.data.Dataset.from_generator(label_generator, label_generator.dtype, label_generator.shape).map(lambda data: tf.expand_dims(data, axis=1))
    return tf.data.Dataset.zip((data_dataset, label_dataset)).shuffle(data_generator.total_samples).batch(1).repeat(), data_generator.total_samples

train_dataset, train_dataset_len = dataset_from_h5_files(train_dirs)
test_dataset, test_dataset_len = dataset_from_h5_files(test_dirs)

# Initialize model and optimizer
model = model.get_model(1024, 3, 2)
optimizer = tf.keras.optimizers.Adam(0.0001)

# https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
def weighted_sparse_categorical_crossentropy(weights):
    weights = tf.keras.backend.variable(weights)
        
    def loss(y_true, y_pred):
        # Convert the input from sparse classes to one-hot encoding
        # TODO: Optimize!
        y_true = tf.keras.backend.cast(y_true, tf.dtypes.int32)
        y_true = tf.keras.backend.one_hot(y_true, 2)

        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)

        # clip to prevent NaN's and Inf's
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        # calc
        loss = y_true * tf.keras.backend.log(y_pred) * weights
        loss = -tf.keras.backend.sum(loss, -1)
        return loss
    
    return loss

loss_fn = weighted_sparse_categorical_crossentropy([0.69843593, 1.02030184])

model.compile(optimizer=optimizer,
              #loss=tf.keras.backend.sparse_categorical_crossentropy,
              #loss='sparse_categorical_crossentropy',
              loss=loss_fn,
              metrics=['sparse_categorical_accuracy'])

try:
    model.load_weights(checkpoint_dir)
    print "Loaded checkpoint: {}".format(checkpoint_dir)
except:
    print "Could not find checkpoint {}, starting from scratch.".format(checkpoint_dir)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir, save_weights_only=True, verbose=0, save_best_only=False)

model.fit_generator(
    train_dataset,
    steps_per_epoch=train_dataset_len,
    validation_data=test_dataset,
    validation_steps=test_dataset_len,
    validation_freq=1,
    epochs=40,
    callbacks=[checkpoint_callback])

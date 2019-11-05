#!/usr/bin/env python3
import sys

# Check arguments (Before imports so we don't have to freaking wait)
if len(sys.argv) > 1:
    input_dir = sys.argv[1]
else:
    print("Usage: {} <dataset dir> <checkpoint_dir> <model_save_dir.h5>".format(sys.argv[0]))
    exit(-1)

if len(sys.argv) > 2:
    checkpoint_dir = sys.argv[2]
else:
    checkpoint_dir = 'tf_ckpts/best.ckpt'

if len(sys.argv) > 3:
    model_save_dir = sys.argv[3]
else:
    model_save_dir = 'saved_model'

from settings import *

import tensorflow as tf
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import h5py
import numpy as np
import model
sys.path.append('./io'); 
from h5_dataset import H5FilesDatasetGenerator
from io_utils import *
sys.path.append('./class_balancing'); 

# Set up the dataset generators 
print("Setting up datasets...")
def dataset_from_h5_files(filenames):
    point_generator = H5FilesDatasetGenerator(filenames, "data")
    label_generator = H5FilesDatasetGenerator(filenames, "label")

    point_dataset = tf.data.Dataset.from_generator(point_generator, point_generator.dtype, point_generator.shape)
    label_dataset = tf.data.Dataset.from_generator(label_generator, label_generator.dtype, label_generator.shape).map(lambda data: tf.expand_dims(data, axis=1))
    combined_dataset = tf.data.Dataset.zip((point_dataset, label_dataset))
    combined_dataset = combined_dataset.shuffle(8000).batch(1).repeat()
    combined_dataset = combined_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return combined_dataset, point_generator.total_samples, label_generator

train_dirs = open_file_list(input_dir, "train_files.txt")
test_dirs = open_file_list(input_dir, "test_files.txt")

train_dataset, train_dataset_len, train_labels = dataset_from_h5_files(train_dirs)
test_dataset, test_dataset_len, _ = dataset_from_h5_files(test_dirs)

# Initialize model and optimizer
print("Building model...")
model = model.get_model(N_POINTS_PER_SAMPLE, N_CLASSES, N_EXTRAS)
optimizer = keras.optimizers.Adam(LEARNING_RATE)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

# Attempt to load a checkpoint TODO: Switch this to an if statement
try: 
    model.load_weights(checkpoint_dir)
    print("Loaded checkpoint: {}".format(checkpoint_dir))
except:
    print("Could not find checkpoint {}, starting from scratch.".format(checkpoint_dir))

# Checkpoint callback for saving 
checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_dir, save_weights_only=True, verbose=0, save_best_only=True)

# Fit the model
#model.fit_generator(
#    train_dataset,
#    steps_per_epoch=train_dataset_len,
#    validation_data=test_dataset,
#    validation_steps=test_dataset_len,
#    validation_freq=1,
#    epochs=N_EPOCHS,
#    callbacks=[checkpoint_callback])

print("Saving model to {}".format(model_save_dir))
model.save(model_save_dir, save_format='tf')

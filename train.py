#!/usr/bin/env python2
import sys

# Check arguments
try:
    input_dir = sys.argv[1]
except:
    print "Usage: {} <dataset dir> <checkpoint_dir>".format(sys.argv[0])
    exit(-1)

try:
    checkpoint_dir = sys.argv[2]
except:
    checkpoint_dir = 'tf_ckpts/best.ckpt'

N_CLASSES = 2
N_POINTS_PER_SAMPLE = 2048
N_CHANNELS = 3
N_EXTRAS = 3
N_EPOCHS = 40
LEARNING_RATE = 0.001

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
from balanced_loss import weighted_sparse_categorical_crossentropy
from balanced_weights import median_frequency_class_weights

# Act like Tensorflow V2.0
tf.enable_eager_execution()

# Set up the dataset generators 
print "Setting up datasets..."
def dataset_from_h5_files(filenames):
    point_generator = H5FilesDatasetGenerator(filenames, "data")
    label_generator = H5FilesDatasetGenerator(filenames, "label")
    extra_generator = H5FilesDatasetGenerator(filenames, "extra")

    point_dataset = tf.data.Dataset.from_generator(point_generator, point_generator.dtype, point_generator.shape)
    label_dataset = tf.data.Dataset.from_generator(label_generator, label_generator.dtype, label_generator.shape).map(lambda data: tf.expand_dims(data, axis=1))
    extra_dataset = tf.data.Dataset.from_generator(extra_generator, extra_generator.dtype, extra_generator.shape)
    combined_dataset = tf.data.Dataset.zip(((point_dataset, extra_dataset), label_dataset))
    combined_dataset = combined_dataset.shuffle(2000).batch(1).repeat()

    return combined_dataset, point_generator.total_samples, label_generator

train_dirs = open_file_list(input_dir, "train_files.txt")
test_dirs = open_file_list(input_dir, "test_files.txt")

train_dataset, train_dataset_len, train_labels = dataset_from_h5_files(train_dirs)
test_dataset, test_dataset_len, _ = dataset_from_h5_files(test_dirs)

# Determine class weights from dataset
print "Determining class weights..."
#class_weights = median_frequency_class_weights(train_labels, N_CLASSES)
class_weights = [0.0765, 1.0]

# Initialize model and optimizer
print "Building model..."
model = model.get_model(N_POINTS_PER_SAMPLE, N_CHANNELS, N_CLASSES, N_EXTRAS)
optimizer = keras.optimizers.Adam(LEARNING_RATE)

# Initialze custom loss function with class weights 
loss_fn = weighted_sparse_categorical_crossentropy(class_weights)

model.compile(optimizer=optimizer,
              #loss='sparse_categorical_crossentropy',
              loss=loss_fn,
              metrics=['sparse_categorical_accuracy'])

# Attempt to load a checkpoint
try:
    model.load_weights(checkpoint_dir)
    print "Loaded checkpoint: {}".format(checkpoint_dir)
except:
    print "Could not find checkpoint {}, starting from scratch.".format(checkpoint_dir)

# Checkpoint for saving 
checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_dir, save_weights_only=True, verbose=0, save_best_only=False)

# Actually fit the model
model.fit_generator(
    train_dataset,
    steps_per_epoch=train_dataset_len,
    validation_data=test_dataset,
    validation_steps=test_dataset_len,
    validation_freq=1,
    epochs=N_EPOCHS,
    callbacks=[checkpoint_callback])

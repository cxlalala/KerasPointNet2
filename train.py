import argparse

parser = argparse.ArgumentParser(description='PointNet++ training script')
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=1e-4)
parser.add_argument('--epochs', dest='n_epochs', type=int, default=40)
parser.add_argument('--points_per_sample', dest='n_points_per_sample', type=int, default=2048)
parser.add_argument('--n_classes', dest='n_classes', type=int, default=2)
parser.add_argument('--shuf_buf_size', dest='shuf_buf_size', type=int, default=3000)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1)
parser.add_argument('--model_save_dir', dest='model_save_dir', type=str, default='./saved_model/')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', type=str, default='./tf_ckpts/best.ckpt')
parser.add_argument('--log_dir', dest='log_dir', type=str, default='./logs/')
parser.add_argument('--dataset_subset', dest='dataset_subset', type=int, default=1)
parser.add_argument('dataset_dir', type=str)
args = parser.parse_args()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import h5py
import numpy as np
import model
from datetime import datetime
from io_utils.h5_dataset import H5FilesDatasetGenerator
from io_utils.misc import open_file_list

# Set up the dataset generators 
print("Setting up datasets...")
def dataset_from_h5_files(filenames):
    point_generator = H5FilesDatasetGenerator(filenames, "data")
    label_generator = H5FilesDatasetGenerator(filenames, "label")

    point_dataset = tf.data.Dataset.from_generator(point_generator, point_generator.dtype, point_generator.shape)
    label_dataset = tf.data.Dataset.from_generator(label_generator, label_generator.dtype, label_generator.shape)

    label_dataset = label_dataset.map(lambda data: tf.expand_dims(data, axis=1))

    combined_dataset = tf.data.Dataset.zip((point_dataset, label_dataset))
    combined_dataset = combined_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    combined_dataset = combined_dataset.shuffle(args.shuf_buf_size).batch(args.batch_size).repeat()
    combined_dataset = combined_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return combined_dataset, point_generator.total_samples, label_generator

train_dirs = open_file_list(args.dataset_dir, "train_files.txt")
test_dirs = open_file_list(args.dataset_dir, "test_files.txt")

train_dataset, train_dataset_len, train_labels = dataset_from_h5_files(train_dirs)
test_dataset, test_dataset_len, _ = dataset_from_h5_files(test_dirs)

# Initialize model and optimizer
print("Building model...")
model = model.get_model(args.n_points_per_sample, args.n_classes)
optimizer = tf.keras.optimizers.Adam(args.learning_rate)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

# Attempt to load a checkpoint TODO: Switch this to an if statement
print("Loading checkpoints...")
try: 
    model.load_weights(args.checkpoint_dir)
    print("Loaded checkpoint: {}".format(args.checkpoint_dir))
except:
    print("Could not find checkpoint {}, starting from scratch.".format(args.checkpoint_dir))

# Checkpoint callback for saving 
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(args.checkpoint_dir, save_weights_only=True, verbose=0, save_best_only=True)

# Tensorboard callback
log_dir = os.path.join(args.log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch = 2, update_freq=10)

# Fit the model
model.fit_generator(
    train_dataset,
    steps_per_epoch=train_dataset_len // args.dataset_subset,
    validation_data=test_dataset,
    validation_steps=test_dataset_len // args.dataset_subset,
    validation_freq=1,
    epochs=args.n_epochs,
    callbacks=[checkpoint_callback, tensorboard_callback])

print("Saving model to {}".format(args.model_save_dir))
model.save(args.model_save_dir, save_format='tf')

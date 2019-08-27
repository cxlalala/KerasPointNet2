#!/usr/bin/env python2
import tensorflow as tf
import sys
sys.path.append('./io'); from h5_dataset import dataset_from_h5_files

tf.enable_eager_execution()
dataset = dataset_from_h5_files(sys.argv[1:])

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(1024, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1024, activation='softmax')
])

optimizer=tf.keras.optimizers.Adam(0.01)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit_generator(dataset.batch(1))

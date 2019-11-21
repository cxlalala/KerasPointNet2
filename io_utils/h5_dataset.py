#!/usr/bin/env python2
import tensorflow as tf
import h5py
import numpy as np

# TODO:
# Dataset 'output' is tightly coupled to the order of items in the HDF5.
# This MUST be rearranged to be more flexible. Maybe pass in a hashmap with
# that maps dataset->index? But it also needs to support nested inputs... Hmm...

# Stream batches from each file
class H5FilesDatasetGenerator:
    def __init__(self, h5_files, dataset):
        self.h5_files = h5_files
        self.dataset = dataset
        self._evaluate_files()

    # Yields an array containing the batches from all files
    def datasets(self):
        for h5_filename in self.h5_files:
            with h5py.File(h5_filename, "r") as h5_file:
                yield h5_file[self.dataset], h5_filename

    # Count number of total samples, check dtype compat, shape compat,
    # assign member variables for each.
    def _evaluate_files(self):
        self.shape = None
        self.dtype = None
        self.total_samples = 0
        for dataset, filename in self.datasets():
            self.total_samples += dataset.shape[0]
            tmp_shape = dataset.shape[1:]
            tmp_dtype = dataset.dtype
            if self.shape is not None and self.dtype is not None:
                if self.shape != tmp_shape:
                    raise ValueError("Shapes do not match! Offender: {}".format(filename))
                if self.dtype != tmp_dtype:
                    raise ValueError("Dtypes do not match! Offender: {}".format(filename))
            else:
                self.shape = tmp_shape
                self.dtype = tmp_dtype

    def __call__(self):
        for dataset, _ in self.datasets(): 
            for sample in dataset:
                yield sample

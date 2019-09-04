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
    def __init__(self, h5_files, mappings):
        self.h5_files = h5_files
        self.mappings=mappings
        self._evaluate_files()

    # Yields an array containing the batches from all files
    def data_batches(self):
        for h5_filename in self.h5_files:
            with h5py.File(h5_filename, "r") as h5_file:
                yield [mapping(h5_file[dataset]) if mapping is not None else h5_file[dataset] for dataset, mapping in self.mappings]

    # Count number of total samples, check dtype compat, shape compat,
    # assign member variables for each.
    def _evaluate_files(self):
        self.shapes = None
        self.dtypes = None
        self.total_samples = 0
        for datasets in self.data_batches():
            self.total_samples += datasets[0].shape[0]
            tmp_shapes = tuple([dataset.shape[1:] for dataset in datasets])
            tmp_dtypes = tuple([dataset.dtype for dataset in datasets])
            if self.shapes is not None and self.dtypes is not None:
                if self.shapes != tmp_shapes:
                    raise ValueError("Shapes do not match!")
                if self.dtypes != tmp_dtypes:
                    raise ValueError("Dtypes do not match!")
            else:
                self.shapes = tmp_shapes
                self.dtypes = tmp_dtypes

    def __call__(self):
        for streams in self.data_batches(): 
            for sample in zip(*streams):
                yield sample

# Get lists of shapes and types for datasets in this h5 file
def shapes_and_types(filename, customshapes):
    with h5py.File(filename, "r") as h5_file:
        types = [h5_file[dataset].dtype for dataset in h5_file]
        shapes = [customshapes[dataset] if customshapes[dataset] else tf.TensorShape(h5_file[dataset].shape[1:]) for dataset in h5_file]
    return (shapes, types)

# Return a tensorflow dataset comprising the h5 files in the input list
def dataset_from_h5_files(h5_files, mappings, customshapes):
    shapes, types = shapes_and_types(h5_files[0], customshapes)
    generator = H5FilesDatasetGenerator(h5_files, mappings)

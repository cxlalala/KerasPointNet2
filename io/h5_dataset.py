#!/usr/bin/env python2
import tensorflow as tf
import h5py

# Stream batches from each file
class H5FilesDatasetGenerator:
    def __init__(self, h5_files):
        self.h5_files = h5_files
        self.current_file = None

    def __call__(self):
        for h5_filename in self.h5_files:
            self.current_file = h5_filename
            with h5py.File(h5_filename, "r") as h5_file:
                for batches in zip(*[h5_file[dataset] for dataset in h5_file]):
                    yield batches

# Get lists of shapes and types for datasets in this h5 file
def shapes_and_types(filename):
    with h5py.File(filename, "r") as h5_file:
        types = [h5_file[dataset].dtype for dataset in h5_file]
        shapes = [tf.TensorShape(h5_file[dataset].shape[1:]) for dataset in h5_file]
    return (shapes, types)

# Return a tensorflow dataset comprising the h5 files in the input list
def dataset_from_h5_files(h5_files):
    shapes, types = shapes_and_types(h5_files[0])
    generator = H5FilesDatasetGenerator(h5_files)
    return tf.data.Dataset.from_generator(generator, tuple(types), tuple(shapes))

# Test it out!
if __name__ == '__main__':
    import sys
    tf.enable_eager_execution()
    dataset = dataset_from_h5_files(sys.argv[1:])
    for batch in dataset.batch(1000):
        print batch

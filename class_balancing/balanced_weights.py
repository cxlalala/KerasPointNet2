#!/usr/bin/env python2
import numpy as np

def median_frequency_class_weights(labels_generator, n_classes):
    shape_counts = np.empty(n_classes)
    
    for label_sample in labels_generator():
        shape_counts = np.vstack((shape_counts, np.bincount(label_sample, minlength=n_classes)))
    
    medians = np.median(shape_counts, axis=0)
    
    n_samples = shape_counts.shape[0]
    frequencies = np.sum(shape_counts, axis=0) / n_samples
    class_weights = medians / frequencies
    return class_weights

if __name__ == "__main__":
    import sys
    sys.path.append('./io'); 
    from h5_dataset import H5FilesDatasetGenerator
    from io_utils import *
    N_CLASSES = 2
    train_dirs = open_file_list("../datasets/net_input/", "train_files.txt")
    labels_generator = H5FilesDatasetGenerator(train_dirs, "label")
    print median_frequency_class_weights(labels_generator, N_CLASSES)

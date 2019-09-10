#!/usr/bin/env python2
import sys
sys.path.append('./io'); 
from h5_dataset import H5FilesDatasetGenerator
from io_utils import *
import numpy as np

N_CLASSES = 2
train_dirs = open_file_list("../datasets/net_input/", "train_files.txt")
labels_generator = H5FilesDatasetGenerator(train_dirs, "label")

shape_counts = np.empty(N_CLASSES)

for label_sample in labels_generator():
    shape_counts = np.vstack((shape_counts, np.bincount(label_sample)))

medians = np.median(shape_counts, axis=0)

contributions = medians / (np.sum(shape_counts, axis=0) / shape_counts.shape[0])

print contributions

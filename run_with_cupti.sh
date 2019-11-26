#!/usr/bin/env bash
#LD_LIBRARY_PATH=/usr/local/cuda-10.0/extras/CUPTI/lib64/ ./train.py /media/storage/datasets/net_input/
LD_LIBRARY_PATH=/usr/local/cuda-10.0/extras/CUPTI/lib64/ ./eval.py /media/storage/datasets/depthmap_test/13231.Geo-{Top,Bottom,Left,Right}.ltiDepthMapStream /media/storage/datasets/net_output ./saved_model

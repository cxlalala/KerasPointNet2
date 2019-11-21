from tensorflow import keras
import sys
sys.path.append('./layers'); from layers import *

def get_model(n_points, n_classes):
    input_xyz = keras.layers.Input(shape=(n_points, 3))
    l0_xyz = input_xyz
    l0_points = None
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1, nsample=32, mlp=[32, 32, 64])
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=256, radius=0.2, nsample=32, mlp=[64, 64, 128])
    #l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=64, radius=0.4, nsample=32, mlp=[128, 128, 256])
    #l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=16, radius=0.8, nsample=32, mlp=[256, 256, 512])
    #l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256, 256])
    #l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256, 256])
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256, 128])
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, mlp=[128, 128, 128])

    net = keras.layers.Conv1D(128, 1, activation='relu')(l0_points)
    net = keras.layers.Dropout(0.5)(net)
    net = keras.layers.Conv1D(n_classes, 1, activation=None)(net)
    net = keras.layers.Softmax()(net)

    return keras.models.Model(inputs=input_xyz, outputs=net)

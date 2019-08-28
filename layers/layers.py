import sys
import tensorflow as tf
sys.path.append('./kernels/grouping'); from grouping import *
sys.path.append('./kernels/interpolation'); from interpolation import *
sys.path.append('./kernels/sampling'); from sampling import *

def sample_and_group(npoint, radius, nsample, xyz, points):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''
    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
    idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1]) # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz

def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, or mlp[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    # Sample and Grouping
    new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points)

    # Point Feature Embedding
    for num_out_channel in mlp:
        new_points = tf.keras.layers.Conv2D(num_out_channel, [1, 1], data_format='channels_last')(new_points)

    # Pooling in Local Regions
    new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
    new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints)

    return new_xyz, new_points, idx

def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    dist, idx = three_nn(xyz1, xyz2)
    dist = tf.maximum(dist, 1e-10)
    norm = tf.reduce_sum((1.0/dist), axis=2, keep_dims=True)
    norm = tf.tile(norm,[1,1,3])
    weight = (1.0 / dist) / norm
    interpolated_points = three_interpolate(points2, idx, weight)
    
    if points1 is not None:
        new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
    else:
        new_points1 = interpolated_points
    
    new_points1 = tf.expand_dims(new_points1, 2)
    
    for num_out_channel in mlp:
        new_points1 = tf.keras.layers.Conv2D(num_out_channel, [1,1])(new_points1)
    
    new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
    return new_points1

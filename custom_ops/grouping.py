import tensorflow as tf
from tensorflow.python.framework import ops
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
grouping_module=tf.load_op_library(os.path.join(BASE_DIR, 'grouping_so.so'))

def query_ball_point(radius, nsample, xyz1, xyz2):
    '''
    Input:
        radius: float32, ball search radius
        nsample: int32, number of points selected in each ball region
        xyz1: (batch_size, ndataset, 3) float32 array, input points
        xyz2: (batch_size, npoint, 3) float32 array, query points
    Output:
        idx: (batch_size, npoint, nsample) int32 array, indices to input points
        pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
    '''
    return grouping_module.query_ball_point(xyz1, xyz2, radius, nsample)
ops.NoGradient('QueryBallPoint')

def group_point(points, idx):
    '''
    Input:
        points: (batch_size, ndataset, channel) float32 array, points to sample from
        idx: (batch_size, npoint, nsample) int32 array, indices to points
    Output:
        out: (batch_size, npoint, nsample, channel) float32 array, values sampled from points
    '''
    return grouping_module.group_point(points, idx)

@tf.RegisterGradient('GroupPoint')
def _group_point_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    return [grouping_module.group_point_grad(points, idx, grad_out), None]

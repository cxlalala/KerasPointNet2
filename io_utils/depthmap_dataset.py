#!/usr/bin/env python3
from struct import unpack, calcsize
import numpy as np

def _yield_depthmap_rows(path):
    """
    Given the path to a depthmap, for each row yield a tuple consisting of:
    depth_values(numpy), fov_start, fov_step, fov_step_count, fov_start_idx, fov_end_idx, background_value, length_position
    """
    header_packing_format = '=hhhhhhhiIxxxxxxxxxx' # Depthmap header
    header_length = calcsize(header_packing_format)
    with open(path, "rb", buffering=2048) as f:
        while True:
            # Read length header
            length_bytes = f.read(4)

            # Detect EOF
            if len(length_bytes) != 4:
                break

            # Unpack entry length header for ltiStream
            length, = unpack('i', length_bytes)

            # Detect file-end marker
            if length == -1:
                break

            # Read and unpack depthmap-specific header
            header_bytes = f.read(header_length)
            header_unpacked = unpack(header_packing_format, header_bytes)
            depth, fov_start, fov_step, fov_step_count, fov_start_idx, fov_end_idx, background_value, length_position, tracking_index = header_unpacked

            # Sanity check
            assert(fov_end_idx - fov_start_idx <= fov_step_count) 

            # Read the strip
            strip_bytes = f.read(length - header_length)
            array_length = (fov_end_idx - fov_start_idx)

            # Convert to numpy
            depth_values = np.frombuffer(strip_bytes, dtype=np.uint16, count=array_length)

            yield (depth_values, fov_start, fov_step, fov_step_count, fov_start_idx, fov_end_idx, background_value, length_position)


def _row_to_3d(row, fov_start, fov_step, fov_start_idx, fov_end_idx, z, orientation):
    """
    Convert a row to a 3D pointcloud.
    `orientation`: Whether this depthmap is top-bottom aligned (True) or left-right aligned (False)
    `z`: Z position for this entire row (assuming row's depth and fov_position represent x and y)
    """
    # Z values repeated for each point
    z_values = np.repeat(z, len(row))

    # Step across the board
    step_values = np.arange(fov_start_idx * fov_step, fov_end_idx * fov_step, fov_step) + fov_start

    # Concatenate the values from each array depthwise (based on orientation)
    if orientation:
        points = np.dstack((step_values, row, z_values))[0]
    else:
        points = np.dstack((row, step_values, z_values))[0]

    return points


def yield_sampled_chunks(top_path, bottom_path, left_path, right_path, n_rows_per_chunk, n_points_per_chunk):
    """
    Provided a path for each depthmap, yield a sub-pointcloud consisting of 
    `n_rows_per_chunk `rows from each face with a randomly-sampled for a total of n_points_per_chunk points.
    """
    bottom_face = _yield_depthmap_rows(bottom_path)
    top_face = _yield_depthmap_rows(top_path)
    left_face = _yield_depthmap_rows(left_path)
    right_face = _yield_depthmap_rows(right_path)
    try:
        row_start_position = 0
    
        while True:
            point_arrays = []
            total_length = 0
    
            # Pull a row out of a face, convert it to 3d, and filter out background values
            def next_row(face, orientation):
                board_slice, fov_start, fov_step, fov_step_count, fov_start_idx, fov_end_idx, background_value, length_position = next(face)
                points = _row_to_3d(board_slice, fov_start, fov_step, fov_start_idx, fov_end_idx, length_position - row_start_position, orientation)
                points = points[board_slice != background_value] #TODO: Find a way to optimize this? It takes a lot of CPU time...
                return (points, length_position)
    
            # Push `n_rows_per_chunk` rows from each face and deposit them in `point_arrays`
            for _ in range(n_rows_per_chunk):
                bottom_row, _ = next_row(bottom_face, True)
                top_row, _ = next_row(top_face, True)
                left_row, _ = next_row(left_face, False)
                right_row, length_position = next_row(right_face, False)
    
                point_arrays.append(bottom_row)
                point_arrays.append(top_row)
                point_arrays.append(left_row)
                point_arrays.append(right_row)
    
                total_length += len(bottom_row)
                total_length += len(top_row)
                total_length += len(left_row)
                total_length += len(right_row)
    
            row_start_position = length_position
    
            # Concatenate the 3d rows from each face into one cloud (without using the copy-heavy np.dstack) 
            cloud = np.ndarray(shape=(total_length, 3))
            position = 0
            for array in point_arrays:
                cloud[position:position+len(array)] = array
                position += len(array)

            assert position == total_length

            # Randomly sample `n_points_per_chunk` from the cloud and yield it
            if len(cloud) != 0:
                #yield cloud
                choices = cloud[np.random.choice(len(cloud), n_points_per_chunk)]
                yield choices

    # Stop yielding values when any of the faces run out of rows
    except StopIteration:
        pass


#N_ROWS_PER_CHUNK=10
#N_POINTS_PER_CHUNK=2048
#
#top_path = "./data/13231.Geo-Top.ltiDepthMapStream"
#bottom_path = "./data/13231.Geo-Bottom.ltiDepthMapStream"
#left_path = "./data/13231.Geo-Left.ltiDepthMapStream"
#right_path = "./data/13231.Geo-Right.ltiDepthMapStream"
#
#import open3d as o3d
#for chunk in yield_sampled_chunks(top_path, bottom_path, left_path, right_path, N_ROWS_PER_CHUNK, N_POINTS_PER_CHUNK):
#    #pass
#    pcd = o3d.geometry.PointCloud()
#    pcd.points = o3d.utility.Vector3dVector(chunk)
#    o3d.visualization.draw_geometries([pcd])

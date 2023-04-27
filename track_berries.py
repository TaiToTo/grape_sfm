import json
import copy
import pickle
import numpy as np
import torch

from sklearn.neighbors import NearestNeighbors

from grape_utils.camera_modules import disp_to_depth
import grape_utils.rotation_matrix as rotation_matrix
from grape_utils.camera_modules import OmniCamera

import sys
import numpy

"""

Track berries in adjacent frames by nearest neighbor.
When frame i, i+1 have M, N berries respectively,
forward_nn_connections: (M, N) array representing wh
backward_nn_connections: (M, N) array
mutual_nn_connections: (M, M) diagonal array. When then No. m berry is mutually nearest neighbor forward and backward, 
(m, m) element of mutual_nn_connections is 1
"""
def make_nn_connections(former_berry_pos, latter_berry_pos):
    nbrs_forward = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(latter_berry_pos.T)
    forward_nn_connection = nbrs_forward.kneighbors_graph(former_berry_pos.T).toarray()

    nbrs_backward = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(former_berry_pos.T)
    backward_nn_connection = nbrs_backward.kneighbors_graph(latter_berry_pos.T).toarray()

    mutual_nn_connections = np.dot(forward_nn_connection, backward_nn_connection)
    return forward_nn_connection, backward_nn_connection, mutual_nn_connections


def berry_clustering_nn(world_coord_list, extrinsic_idx_range, cluster_start_idx, cluster_thresh=5):
    whole_berry_pos_array = np.concatenate(world_coord_list, 1)[:3,:]
    whole_berry_idx_list = [(np.arange(world_coord_list[i].shape[1]) + cluster_start_idx) for i in range(len(world_coord_list))]
    whole_berry_idx_array = np.arange(whole_berry_pos_array.shape[1])
    extrinsic_idx_array = np.concatenate([[i] * (world_coord_list[i].shape[1]) for i in range(len(world_coord_list))])

    max_idx = 0
    for i in range(len(whole_berry_idx_list)):
        whole_berry_idx_list[i] = whole_berry_idx_list[i] + max_idx
        max_idx = whole_berry_idx_list[i].max() + 1

    # forward_connection_list = []
    forward_mutual_connection_tuple_list = []

    for cnt, frame_idx in enumerate(range(extrinsic_idx_range[0], extrinsic_idx_range[-1])):
        former_berry_pos_array = world_coord_list[frame_idx][:3, :].copy()
        latter_berry_pos_array = world_coord_list[frame_idx + 1][:3, :].copy()

        forward_nn_connections, backward_nn_connection, mutual_nn_connections = make_nn_connections(
            former_berry_pos_array, latter_berry_pos_array)

        forward_mutual_connection_tuple_list.append((forward_nn_connections, mutual_nn_connections))




    for cnt_0, extrinsic_idx in enumerate(range(frame_idx_range[0], frame_idx_range[-1])):
        mutual_nn_connections = forward_mutual_connection_tuple_list[cnt_0][1]
        forward_nn_connections = forward_mutual_connection_tuple_list[cnt_0][0]
        
        for cnt_1 in range(mutual_nn_connections.shape[0]):
            if (mutual_nn_connections[cnt_1, cnt_1] == 1):
                current_idx = whole_berry_idx_list[cnt_0][cnt_1]
                next_idx = np.argmax(forward_nn_connections[cnt_1])
                whole_berry_idx_list[cnt_0 + 1][next_idx] = current_idx


    berry_tracked_idx_array = np.concatenate(whole_berry_idx_list, 0)
    tracked_berry_range = np.unique(berry_tracked_idx_array)

    # Berries composing clusters less than cluster_threshold elements as untracked (-1)
    for idx in tracked_berry_range:
        if ((berry_tracked_idx_array == idx).astype(np.int8).sum() < cluster_thresh):
            berry_tracked_idx_array[berry_tracked_idx_array == idx] = -1

    return berry_tracked_idx_array, whole_berry_pos_array, extrinsic_idx_array

"""
Loading local data necessary for tracking berreis.
R0010110_berry_pos_coords.json: manual annotations of berry centers of every frame.
disp_array_R0010110.npy: disp (inverse matrix internally used in deep learning) estimated by monocular depth estimation.
cam_pose_array_R0010110.npy: A sequence of relative camera poses of adjascent frame pairs.
depth_coeffs.py: coefficients for calculating 2d → 3d backprojection with omnidirectional camera.
"""
def get_initial_data():
    with open('data_folder/R0010110_berry_pos_coords.json') as f:
        center_coords = json.load(f)
    cam_pose_array_R0010110 = np.load('data_folder/cam_pose_array_R0010110.npy')
    disp_array = np.load('data_folder/disp_array_R0010110.npy')
    norm_coeffs = np.expand_dims(np.load('data_folder/depth_coeffs.npy'), 0)
    return center_coords, cam_pose_array_R0010110, disp_array, norm_coeffs

"""
# Returns cam_T_list, extrinsic_matrix_list, and world_coord_list.
# * 注:  同様のループを繰り返しているため冗長な可能性あり.　ループをするフレームのインデックスの範囲も一貫していない場合あり．
"""
def get_extrinsic_and_world_coord(camera, extracted_frame_index_range, center_coords, frame_idx_range):
    cam_T_list = []
    extrinsic_matrix_list = []
    world_coord_list = []

    cam_T_list = [cam_pose_array_R0010110[idx] for idx in range(first_frame_idx, last_frame_idx - 1)]
    cam_T_list = [np.eye(4).astype(np.float32)] + cam_T_list
    current_extrinsic_parameter = np.eye(4).astype(np.float32)

    for extrinsic_idx, frame_idx in enumerate(range(first_frame_idx, last_frame_idx)):
        current_extrinsic_parameter = np.dot(current_extrinsic_parameter, cam_T_list[extrinsic_idx])
        extrinsic_matrix_list.append(current_extrinsic_parameter)

    # Converting 3d positions of berries in each camera coordinate to the world coordiante.
    for extrinsic_idx, frame_idx in enumerate(frame_idx_range):

        # Gets the disp of No. frame_idx disp (inverse depth used internally in deep learning code)
        disp = disp_array[frame_idx]

        # Converts the inverse depth to the depth of omnidirectional camera, whose range is from 1 to 100
        _, depth = disp_to_depth(disp, 1, 100)
        depth = np.expand_dims(depth, 0)
        depth = torch.from_numpy(depth)

        # Currently using manual labels, labeled on (960, 960) images, but images used in this code are (480. 480)
        berry_pos_2d = np.array(center_coords['frame' + str(frame_idx + 1) + '.jpg']['berry_pos']) / 2
        img_coord_indexes = camera.get_discrete_img_coord_indexes(berry_pos_2d)
        x_s = camera.backproject_to_camera_coords(depth, img_coord_indexes)
        homogeneous_dimension = torch.from_numpy(np.ones((1, x_s.shape[0])))
        x_s = torch.transpose(x_s, 0, 1)
        homogeneous_x_s = np.concatenate([x_s, homogeneous_dimension])

        # Converts the camera coord to the world coord.
        x_s_world_coord = np.dot(np.linalg.inv(extrinsic_matrix_list[extrinsic_idx]), homogeneous_x_s)
        world_coord_list.append(x_s_world_coord)


    # Calculates 6-DoF parameters of the extrinsic parameters above
    six_dof_camera_param_list = []
    temp_6_dof = np.zeros(6)
    print(extracted_frame_index_range[1:-1])
    for cnt, frame_idx in enumerate(extracted_frame_index_range[1:-1]):
        cnt = cnt + 1
        extrinsic_matrix = extrinsic_matrix_list[cnt]
        R, t = extrinsic_matrix[:3, :3], extrinsic_matrix[:3, 3]
        axis, theta = rotation_matrix.R_to_axis_angle(R)
        six_dof_camera_param_list.append(np.concatenate([axis * theta, t]))

    return extrinsic_matrix_list, world_coord_list, six_dof_camera_param_list


"""

"""
def track_berries(world_coord_list, center_coords, start_frame_idx, last_frame_idx):
    cluster_start_idx = 1 # Indexes of clusters of 3d points start from 1

    tracked_world_coord_list = []
    tracked_berry_idx_list = []
    tracked_points_2d_list = []
    frame_idx_array_list = []

    points_2d = []

    frame_idx_range_temp = range(start_frame_idx, last_frame_idx) # frame index range
    extrinsic_idx_range = range(last_frame_idx - start_frame_idx) # extrinsic index range starts from 0

    # Gets 2d berry positions of the frames.
    for frame_idx in frame_idx_range_temp:
        points_2d.append(center_coords['frame' + str(frame_idx + 1) + '.jpg']['berry_pos'])
    points_2d = np.concatenate(points_2d, 0) / 2

    berry_pos_to_cluster = np.concatenate(world_coord_list, 1)[:3, :][[0, 1, 2]] # Concatenates all the 3d points
    berry_pos_to_cluster = berry_pos_to_cluster.T

    # Makes a list/array storing frame indexes corresponding to
    frame_idx_list_to_cluster = [[start_frame_idx + local_idx] * world_coord_list[local_idx].shape[-1]
                                     for local_idx in range(len(world_coord_list))]
    frame_idx_array_to_track = np.concatenate(frame_idx_list_to_cluster)

    berry_cluster_idx_tracked, _, _ = berry_clustering_nn(world_coord_list, extrinsic_idx_range, cluster_start_idx)

    frame_idx_array_list.append(berry_cluster_idx_tracked)
    # frame_idx_array_list.append(frame_idx_array_to_track)

    idx_of_interest = berry_cluster_idx_tracked != -1

    point_indices = berry_cluster_idx_tracked[idx_of_interest]
    camera_indices = frame_idx_array_to_track[idx_of_interest]
    
    berry_local_cluster_idx_temp = berry_cluster_idx_tracked

    new_berry_idx_range = range(cluster_start_idx, berry_local_cluster_idx_temp.max().astype(np.int8))

    for cluster_idx in new_berry_idx_range:
        selected_idx = berry_local_cluster_idx_temp == cluster_idx
        if (sum(selected_idx.astype(np.int8)) >= 1):
            selected_berry_pos = berry_pos_to_cluster[selected_idx][:, [0, 1, 2]]
            cluster_mean = selected_berry_pos.mean(0)

    local_cluster_idx = berry_cluster_idx_tracked.max().astype(np.int8) + 1

    return berry_cluster_idx_tracked, berry_pos_to_cluster, points_2d, frame_idx_array_list, camera_indices


"""
Merges tracked 3d points in each cluster into a point and prepared initial 3d coordinates and corresponding
indexes needed for bundle adjustment.
"""
def merge_tracked_data(berry_cluster_idx_tracked, berry_pos_to_cluster, points_2d, frame_idx_array_list):

    # Concatenates the tracked data into data with the same length.
    world_points_before_merge = berry_pos_to_cluster
    points_2d_before_merge = points_2d
    berry_indexes_before_merge = berry_cluster_idx_tracked
    camera_indices_before_merge = np.concatenate(frame_idx_array_list)

    world_points = []
    camera_indices = []
    point_indices = []
    # points_2d = []

    # Extracts tracked berry indexes
    unique_berry_index_list = [idx for idx in np.unique(berry_indexes_before_merge) if idx not in [-1, 0]]

    # Calculates a mean of each tracked 3d points cluster and appends them to world_points
    # Coordinates in world_points are the initial 3d points used for bundle adjustment.
    for reassigned_point_idx, unique_berry_idx in enumerate(unique_berry_index_list): # A loop over indexes of clusters of tracked 3d points
        extracted_idx = berry_indexes_before_merge == unique_berry_idx
        extracted_world_coordinates = world_points_before_merge[extracted_idx]
        world_points.append(extracted_world_coordinates.mean(0))

    # Gets indexes of tracked berries and extracts only necessary data
    tracked_unique_berry_index = berry_indexes_before_merge != -1
    world_points = np.stack(world_points)
    camera_indices = camera_indices_before_merge[tracked_unique_berry_index]
    # point_indices = point_indices_before_merge[tracked_unique_berry_index]
    point_indexes = berry_indexes_before_merge[tracked_unique_berry_index]
    points_2d = points_2d_before_merge[tracked_unique_berry_index]


    # Mapping berry indexes to new consecutive indexes
    # * berry_indexes are not consecutive, and coudl be much bigger than the size of total 3d berry points
    unique_point_indexes = np.unique(point_indexes)
    point_indexes_reassigned = np.zeros(point_indexes.shape)

    # Maps a former_point_idx to a reassigned_point_idx, and also maps indexes related (point_indexes )
    for reassigned_point_idx, former_point_idx in enumerate(unique_point_indexes):
        extracted_indices = point_indexes == former_point_idx
        point_indexes_reassigned[extracted_indices] = reassigned_point_idx + 1

    return world_points, point_indexes_reassigned, points_2d


if __name__ == "__main__":
    # Reading data externally
    # Would be replaced with deep learning outputs
    center_coords, cam_pose_array_R0010110, disp_array, norm_coeffs = get_initial_data()


    # Parameter configurations (frame sizes, intrinsic paramters)
    batch_size = 1
    device = torch.device('cpu')
    width = 480
    height = 480
    K = np.array([[427.85318851, 0.0, 241.05120576, 0],
                  [0, 428.12981526, 238.43306185, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
    xi = 2.0796752

    inv_K = np.linalg.inv(K)

    # A class for calcualtion with a caribrated omnidirectional camera model.
    camera_class = OmniCamera(batch_size, height, width, K, inv_K, xi, norm_coeffs, device)

    local_bunch_frame_interval = 50 # How many frames to use for bundle adjustment
    first_frame_idx = 0 # The first frame
    last_frame_idx = first_frame_idx + local_bunch_frame_interval # The last frame

    frame_idx_range = range(first_frame_idx, last_frame_idx)
    extracted_frame_index_range = range(first_frame_idx, first_frame_idx+local_bunch_frame_interval)

    # Prepares extrinsic parameters world coord of 3d positions of berries.
    extrinsic_matrix_list, world_coord_list, six_dof_camera_param_list \
        = get_extrinsic_and_world_coord(camera_class, extracted_frame_index_range, center_coords, frame_idx_range)

    berry_cluster_idx_tracked, berry_pos_to_cluster, \
    points_2d, frame_idx_array_list, camera_indices = track_berries(world_coord_list, center_coords, first_frame_idx, last_frame_idx)

    world_points, point_indexes_reassigned, points_2d = merge_tracked_data(berry_cluster_idx_tracked, berry_pos_to_cluster, points_2d, frame_idx_array_list)


    # Saving data necessary for bundle adjustment.
    # *注: かなり適当な保存方法
    bundle_data = [np.stack(six_dof_camera_param_list), # 6-DoF extrinisc parameters
                   world_points, #
                   camera_indices - first_frame_idx,
                   point_indexes_reassigned,
                   points_2d,
                   list(frame_idx_range)]
                   
    f = open('./data_folder/bundle_data/bundle_data_R0010110_' + str(frame_idx_range[0])+ '_' + str(frame_idx_range[-1])+ '.txt', 'wb')
    pickle.dump(bundle_data, f)
    
   










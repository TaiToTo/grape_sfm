import pickle
import numpy as np
import torch
import os

from grape_utils.grape_bunch_module import PartialBunch

def partial_grape_bundle_adjustment(partial_bunch, frame_index_pair):
    #residual_list = []

    key_frame_interval = camera_params.shape[0] + 1
    n_cameras = camera_params.shape[0] + 1
    n_points = world_points.shape[0]

    num_extrinsics = partial_bunch.n_cameras - 1 # The extrinisc parameters of the first camera is an identity matrix, which is not updated.
    camera_params_initial = partial_bunch.x0[:num_extrinsics * partial_bunch.deg_freedom].reshape((num_extrinsics, partial_bunch.deg_freedom))
    points_3d_initial = partial_bunch.x0[partial_bunch.deg_freedom * num_extrinsics:].reshape((partial_bunch.n_points, 3))

    points_3d_on_sphere, points_2d_on_sphere = partial_bunch.fun_on_sphere(partial_bunch.x0)

    partial_bunch.grape_bundle_adjustment(if_alternate_optimization=False, ftol=0.1)
    #residual_list.append(residual)
    #return partial_bunch

    with open('bundle_simultaneous_0_49.pickle', mode='wb') as f:
        pickle.dump(partial_bunch, f)

if __name__ == "__main__":
    batch_size = 1 # Just a random value for now.
    device = torch.device('cpu') # Much of data are torch tensors in this implemntation.
    width = 480 # Frame width
    height = 480 # Frame height
    min_depth = 1 # Minimum value of depth.
    max_depth = 100 # Maximum value of depth.

    K = np.array([[427.85318851, 0.0, 241.05120576, 0], # The intrinsic parameter for pinhole projection.
                  [0, 428.12981526, 238.43306185, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 1]], dtype=np.float32)

    xi = 2.0796752 # Another intrinsic parameter.
    inv_K = np.linalg.inv(K) # An inverse matrix of K.

    frame_index_pair = (0, 49)
    first_frame_index, last_frame_index = frame_index_pair

    print("Configuring partial grape bundle adjustment from frame {} to {}".format(first_frame_index, last_frame_index))

    #bundle_adjustment_mode = 'alternate'
    bundle_adjustment_mode = 'simultaneous'

    result_dir = './result_dir'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    print("Results will be saved to {}".format(result_dir))
    print()

    print("Loading depth data")
    print("Loading coefficients of backprojections with unified omnidirectional camera model")
    print()
    disp_array = np.load('./data_folder/disp_array_R0010110.npy')  # Inverse depth estimated by monocular depth estimation.
    norm_coeffs = np.expand_dims(np.load('./data_folder/depth_coeffs.npy'), 0)  # Cofficients of depths (x norms).
    f = open("./data_folder/bundle_data_R0011010_" + str(first_frame_index) + "_" + str(last_frame_index) + ".txt", "rb")

    print("Loading initial 2d berrycoordinates")
    print("Loading initial extrinsic paramters")
    print("Loading initial 3d berry coordinates")
    print("Loading tracked berry indexes")
    camera_params, world_points, camera_indices, point_indices, points_2d, extrinsic_indexes = pickle.load(f)
    point_indices = (point_indices - 1).astype(int) # A small adjustment for indexing tensors or ararys.


    # Initializing a class for a partial grape bunch.
    partial_bunch = PartialBunch(batch_size, height, width, K, inv_K, xi, norm_coeffs, device, disp_array, min_depth, max_depth,
                                 camera_params, world_points, camera_indices, point_indices, points_2d, extrinsic_indexes)
    print("A partial grape class configurated")

    # Execute bundle adjustment.
    print("Executing bundle adjustment of the partial grape bunch")
    partial_grape_bundle_adjustment(partial_bunch, frame_index_pair)

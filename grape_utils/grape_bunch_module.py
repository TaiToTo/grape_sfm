import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.optimize import least_squares

from .camera_modules import  disp_to_depth, OmniCamera, convert_6dof_to_extrinsic

class PartialBunch(nn.Module):
    def __init__(self, batch_size, height, width, K, inv_K, xi, norm_coeffs, device, disp, min_depth, max_depth,
                 camera_params, world_points, camera_indices, point_indices, points_2d, extrinsic_indexes, eps=1e-7):
        super(PartialBunch, self).__init__()

        self.height = height
        self.width = width

        self.camera_class = OmniCamera(batch_size, height, width, K, inv_K, xi, norm_coeffs, device, )

        self.camera_indices = camera_indices
        self.point_indices = point_indices
        self.points_2d = points_2d
        self.extrinsic_indexes = extrinsic_indexes

        # Setting the numbers of cameras and points on world coordinate in this key frame class.
        self.n_cameras = camera_params.shape[0] + 1
        self.n_points = world_points.shape[0]
        self.deg_freedom = camera_params.shape[1]

        self.baseline = camera_params[0][3:]

        self.x_camera = camera_params.ravel()
        self.x_world = world_points.ravel()

        self.x0 = np.hstack((self.x_camera, self.x_world))
        self.x = np.hstack((self.x_camera, self.x_world))

        self.disp_list = [disp[i] for i in extrinsic_indexes]
        self.depth_list = [disp_to_depth(self.disp_list[i], min_depth, max_depth)[0] for i in
                           range(len(extrinsic_indexes))]
        self.min_depth = torch.ones(self.depth_list[0].shape) * min_depth
        self.extrinsic_matrix_list = []

        self.residual_initial = self.fun(self.x0)
        self.residual = self.fun(self.x)

    def project_on_sphere(self, points_3d, camera_params):
        """Rotate points by given rotation vectors.
        Rodrigues' rotation formula is used.
        """
        points_rotated_list = []
        points_3d_on_sphere_list = []

        extrinsic_parameter_list = self.get_extrinsic_list(camera_params)

        for local_camera_index in range(self.n_cameras):
            camera_indexes_projected = self.camera_indices == local_camera_index
            point_indexes_projected = self.point_indices[camera_indexes_projected].astype(np.int)
            points_3d_projected = points_3d[point_indexes_projected]
            homogenous_column = np.ones(points_3d_projected.shape[0])[:, np.newaxis]
            points_3d_homogeneous = np.concatenate([points_3d_projected, homogenous_column], 1)
            points_rotated = np.dot(extrinsic_parameter_list[local_camera_index], points_3d_homogeneous.T)
            points_3d_on_sphere_list.append(self.camera_class.projection_on_sphere(points_rotated))

        points_3d_on_sphere = torch.cat(points_3d_on_sphere_list)
        return points_3d_on_sphere

    def backproject_on_sphere(self):
        """Rotate points by given rotation vectors.

        Rodrigues' rotation formula is used.
        """

        points_2d_on_sphere_list = []

        for local_camera_index in range(self.n_cameras):
            point_indexes_projected = self.camera_indices == local_camera_index
            points_2d_projected = self.points_2d[point_indexes_projected]
            img_coord_indexes = self.get_discrete_img_coord_indexes(points_2d_projected)
            img_points_projected_on_sphere = self.camera_class.backproject_to_camera_coords(self.min_depth,
                                                                                            img_coord_indexes)
            points_2d_on_sphere_list.append(img_points_projected_on_sphere)

        points_2d_on_sphere = torch.cat(points_2d_on_sphere_list)
        return points_2d_on_sphere

    def get_discrete_img_coord_indexes(self, berry_pos_2d):
        berry_pos_2d_discrete = np.round(berry_pos_2d)
        berry_pos_indexes = torch.tensor([(berry_pos_2d_discrete[i][0] + self.height * berry_pos_2d_discrete[i][1]) for i in
                                          range(berry_pos_2d_discrete.shape[0])])
        return berry_pos_indexes.long()

    def fun_on_sphere(self, params):
        """Compute residuals.
        `params` contains camera parameters and 3-D coordinates.
        """
        number_of_extrinsic_parameters = self.n_cameras - 1
        camera_params = params[:number_of_extrinsic_parameters * self.deg_freedom].reshape(
            (number_of_extrinsic_parameters, self.deg_freedom))
        points_3d = params[self.deg_freedom * number_of_extrinsic_parameters:].reshape((self.n_points, 3))
        points_3d_on_sphere = self.project_on_sphere(points_3d, camera_params)
        points_2d_on_sphere = self.backproject_on_sphere()

        return points_3d_on_sphere, points_2d_on_sphere

    def fun_camera(self, params):
        """Compute residuals.
        `params` contains camera parameters and 3-D coordinates.
        """
        number_of_extrinsic_parameters = self.n_cameras - 1
        camera_params = params.reshape((number_of_extrinsic_parameters, self.deg_freedom))
        points_3d = self.x_world.reshape((self.n_points, 3))
        points_3d_on_sphere = self.project_on_sphere(points_3d, camera_params)
        points_2d_on_sphere = self.backproject_on_sphere()
        residuals = points_3d_on_sphere.numpy() - points_2d_on_sphere.numpy()
        return residuals.ravel()

    def fun_world(self, params):
        """Compute residuals.
        `params` contains camera parameters and 3-D coordinates.
        """
        number_of_extrinsic_parameters = self.n_cameras - 1
        camera_params = self.x_camera.reshape((number_of_extrinsic_parameters, self.deg_freedom))
        points_3d = params.reshape((self.n_points, 3))
        points_3d_on_sphere = self.project_on_sphere(points_3d, camera_params)
        points_2d_on_sphere = self.backproject_on_sphere()
        residuals = points_3d_on_sphere.numpy() - points_2d_on_sphere.numpy()
        return residuals.ravel()

    def fun(self, params):
        """Compute residuals.
        `params` contains camera parameters and 3-D coordinates.
        """
        number_of_extrinsic_parameters = self.n_cameras - 1
        camera_params = params[:number_of_extrinsic_parameters * self.deg_freedom].reshape(
            (number_of_extrinsic_parameters, self.deg_freedom))
        transition_params_flattened = camera_params[:, 3:].ravel()
        points_3d = params[self.deg_freedom * number_of_extrinsic_parameters:].reshape((self.n_points, 3))
        points_3d_on_sphere = self.project_on_sphere(points_3d, camera_params)
        points_2d_on_sphere = self.backproject_on_sphere()
        residuals = points_3d_on_sphere.numpy() - points_2d_on_sphere.numpy()  # +
        return residuals.ravel()

    def grape_bundle_adjustment(self, if_alternate_optimization=True, ftol=1e-4):
        # Setting initial variables of bundle adjustment.

        t0 = time.time()

        if (if_alternate_optimization):
            x_initial_camera = self.x_camera
            x_initial_world = self.x_world
            res = least_squares(self.fun_camera, x_initial_camera,
                                jac_sparsity=None, verbose=2, x_scale='jac', ftol=ftol, method='trf', loss='huber')
            self.x_camera = res['x']
            self.x = np.hstack((self.x_camera, self.x_world))

            res = least_squares(self.fun_world, x_initial_world,
                                jac_sparsity=None, verbose=2, x_scale='jac', ftol=ftol, method='trf', loss='huber')
            self.x_world = res['x']
            self.x = np.hstack((self.x_camera, self.x_world))
            if_fix_camera_params = 0

            self.residual = self.fun(self.x)
        else:
            x_initial = self.x
            res = least_squares(self.fun, x_initial,
                                jac_sparsity=None, verbose=2, x_scale='jac', ftol=ftol, method='trf', loss='huber')
            self.x_camera = res['x']
            self.x = res['x']
            self.residual = self.fun(self.x)


        t1 = time.time()
        print("Optimization took {0:.0f} seconds".format(t1 - t0))
        #return self.residual

    def get_extrinsic_list(self, camera_params):
        extrinsic_matrix_list = [np.eye(4)]
        for i in range(self.n_cameras - 1):
            T = convert_6dof_to_extrinsic(camera_params[i])
            extrinsic_matrix_list.append(T)
        return extrinsic_matrix_list

    def plot_residual(self, residual_range=0.1, save_dir='.'):
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 2, 1)
        plt.title('Initial residuals', fontsize=20)
        plt.ylim([-residual_range, residual_range])
        plt.plot(self.residual_initial)
        plt.subplot(1, 2, 2)
        plt.title('Current residuals', fontsize=20)
        plt.ylim([-residual_range, residual_range])
        plt.plot(self.residual)
        plt.show()
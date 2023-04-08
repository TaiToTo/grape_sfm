import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import rotation_matrix

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def convert_6dof_to_extrinsic(params_6dof):
    T = np.zeros((4, 4))
    rot_vec = params_6dof[:3]
    theta = np.linalg.norm(rot_vec)
    unit_axis = rot_vec / theta
    #R = rotation_matrix.R_axis_angle(unit_axis, theta)
    R = rotation_matrix.R_axis_angle(unit_axis, theta)
    T[:3, :3] = R
    T[:3, 3] = params_6dof[3:]
    return T

class OmniCamera(nn.Module):
    def __init__(self, batch_size, height, width, K, inv_K, xi, norm_coeffs, device,  eps=1e-7):
        super(OmniCamera, self).__init__()
        self.cnt = 0
        #self.opt = opt
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps
        self.K = torch.from_numpy(np.expand_dims(K, 0))
        self.inv_K = torch.from_numpy(np.expand_dims(inv_K, 0))
        self.xi = torch.tensor(xi).to(device)
        self.device = device
        self.t_xi = torch.zeros((batch_size, 3, height*width)).to(device)
        self.t_xi[:, 2, :] = float(xi)
        
        self.norm_coeffs = torch.from_numpy(norm_coeffs)
        
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width), requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)

        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1), requires_grad=False)

    def backproject_to_camera_coords(self, depth, img_coord_indexes=None):
        x_s_item_1 = (depth*torch.unsqueeze(self.norm_coeffs, 1)).view((self.batch_size, 1, -1))*torch.matmul(self.inv_K[ :, :3, :3], self.pix_coords)
        x_s_item_2  = (depth.view((self.batch_size, 1, -1)) - 1)*self.t_xi         
        x_s = x_s_item_1 - x_s_item_2  
        
        if(img_coord_indexes is not None):
            return x_s[:, :, img_coord_indexes]
        
        return x_s

    def get_discrete_img_coord_indexes(self, berry_pos_2d):
        berry_pos_2d_discrete = np.round(berry_pos_2d)
        berry_pos_indexes = torch.tensor([(berry_pos_2d_discrete[i][0] + self.height*berry_pos_2d_discrete[i][1]) for i in range(berry_pos_2d_discrete.shape[0])])
        return berry_pos_indexes.long()

    def projection_on_sphere(self, points_rotated):
        x_s = torch.from_numpy(points_rotated[:3, :].T)
        x = x_s.clone()
        x[:, 2] = x[:, 2] - self.xi
        x_norm = torch.unsqueeze(torch.norm(x,  2, 1), 1)
        x_s_dash = x_s/x_norm
        x_s_dash[:, 2] = x_s_dash[:, 2] + self.xi* (x_norm[:, 0] - 1)/x_norm[:, 0] 
        return x_s_dash
    
    def projection_on_img(self, points_3d_on_sphere):
        points_3d_on_img = torch.matmul(self.K.double()[:, :3, :3], points_3d_on_sphere.T)
        points_3d_on_img = points_3d_on_img[0, :, :] / points_3d_on_img[0, 2, :] 
        return points_3d_on_img
    
    def backproject_to_camera_coords(self, depth, img_coord_indexes=None):
        x_s_item_1 = (depth*torch.unsqueeze(self.norm_coeffs, 1)).view((self.batch_size, 1, -1))*torch.matmul(self.inv_K[ :, :3, :3], self.pix_coords)
        x_s_item_2  = (depth.view((self.batch_size, 1, -1)) - 1)*self.t_xi         
        x_s = x_s_item_1 - x_s_item_2 
        
        if(img_coord_indexes is not None):
            return x_s[0, :, img_coord_indexes].T
        return x_s
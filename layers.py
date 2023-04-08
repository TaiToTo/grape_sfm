# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


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

def disp_to_depth_modified(disp, min_depth, max_depth, device):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    #print(torch.ones(disp.shape).dtype, disp.dtype)
    #print("min_disp: {}, max_disp: {}".format(min_disp, max_disp))
    #print(disp.min(), disp.max())
    scaled_disp = min_disp + (max_disp - min_disp) * (torch.ones(disp.shape).to(device) - disp)
    #scaled_disp = min_disp + (max_disp - min_disp) * (disp)
    depth = 1 / scaled_disp
    return depth

'''
def disp_to_depth_modified(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_depth = min_depth + (max_depth - min_depth) * disp
    return scaled_depth
'''

def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.cnt = 0

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points
    
    



class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()
        self.cnt = 0
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords
 
    #OmniDIBR(self.opt.batch_size, h, w, self.xi, self.device, self.opt)
    
class OmniDIBR(nn.Module):
    def __init__(self, batch_size, height, width, xi, device, opt, eps=1e-7):
        super(OmniDIBR, self).__init__()
        self.cnt = 0
        self.opt = opt
        self.batch_size = batch_size
        #self.batch_size = 8
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps
        self.xi = torch.tensor(xi).to(device)
        self.device = device
        self.t_xi = torch.zeros((batch_size, 3, height*width)).to(device)
        self.t_xi[:, 2, :] = float(xi)
        
        self.min_depth = 1
        self.max_depth = 100
        self.export_interval = 500
        
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width), requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)

        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1), requires_grad=False)
        
    # A fucntion for mapping from a p_t to p_t'
    def pixel_index_mapping(self, a, b):
        b_idx_mapper = b + torch.unsqueeze(torch.arange(b.shape[0]), 1).to(self.device) * b.shape[1]
        mapped_a  = a.permute(1, 0, 2).reshape(3, -1)[:, b_idx_mapper.reshape(-1)].reshape(a.shape[1], a.shape[0], a.shape[2]).permute(1, 0, 2)
        return mapped_a

    def forward(self, depth, inv_K, K, norm_coeffs, norm_solution_area, T, batch_idx):
        #print((norm_solution_area==0).shape)
        #print(type(self.batch_size))
        indexes_outside_circle = ((norm_solution_area==0)+(norm_solution_area==-1)).view((int(self.batch_size),  -1))
        #indexes_outside_circle = (norm_solution_area==0).view((8,  -1))
        indexes_outside_circle = indexes_outside_circle[0]

        indexes_inside_circle = (norm_solution_area==1).view((self.batch_size,  -1))
        indexes_inside_circle = indexes_inside_circle[0]
        #print("'indexes_outside_circle: {}, indexes_inside_circle: {}".format(indexes_outside_circle.sum(), indexes_inside_circle.sum()))
        
        #print("Cehckpoint A")
        self.cnt = self.cnt + 1

        depth_inside_circle = depth.view((8, -1))
        depth_inside_circle = depth_inside_circle[:, indexes_inside_circle]

        
        norm_solution_area = torch.unsqueeze(norm_solution_area, 1)
        
        if(self.cnt%1000 == 0):
            print("depth inside circle: min={}, max={}".format(depth_inside_circle[0].min(), depth_inside_circle[0].max()))
            with open(self.opt.log_dir + "/" + self.opt.model_name + '.txt', 'a') as f:
                #print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                #                     sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)), file=f)
                f.write("depth inside circle: min={}, max={}".format(depth_inside_circle[0].min(), depth_inside_circle[0].max()) + '\n')
        
        
        
        min_depth_tensor = torch.ones(depth.shape).to(self.device)*self.min_depth
        max_depth_tensor = torch.ones(depth.shape).to(self.device)*self.max_depth
        
        x_s1_min_item_1 = (min_depth_tensor*torch.unsqueeze(norm_coeffs, 1)).view((self.batch_size, 1, -1))*torch.matmul(inv_K[:, :3, :3],self.pix_coords)
        x_s1_min_item_2  = (min_depth_tensor.view((self.batch_size, 1, -1)) - 1)*self.t_xi         
        x_1_min = x_s1_min_item_1 - x_s1_min_item_2 # x_1_min(p_t)
        x_s1_max_item_1 = (max_depth_tensor*torch.unsqueeze(norm_coeffs, 1)).view((self.batch_size, 1, -1))*torch.matmul(inv_K[:, :3, :3],self.pix_coords)
        x_s1_max_item_2  = (max_depth_tensor.view((self.batch_size, 1, -1)) - 1)*self.t_xi         
        x_1_max = x_s1_max_item_1 - x_s1_min_item_2 # x_1_max(p_t)
        
        
        x_s2_min_item_1 = (min_depth_tensor*torch.unsqueeze(norm_coeffs, 1)).view((self.batch_size, 1, -1))*torch.matmul(inv_K[:, :3, :3],self.pix_coords)
        x_s2_min_item_2  = (min_depth_tensor.view((self.batch_size, 1, -1)) - 1)*self.t_xi         
        x_2_min = x_s2_min_item_1 - x_s2_min_item_2 # x_2_min(p_t')
        x_s2_max_item_1 = (max_depth_tensor*torch.unsqueeze(norm_coeffs, 1)).view((self.batch_size, 1, -1))*torch.matmul(inv_K[:, :3, :3],self.pix_coords)
        x_s2_max_item_2  = (max_depth_tensor.view((self.batch_size, 1, -1)) - 1)*self.t_xi         
        x_2_max = x_s2_max_item_1 - x_s2_min_item_2 # x_2_max(p_t')
        
        
        x_s1_item_1 = (depth*torch.unsqueeze(norm_coeffs, 1)).view((self.batch_size, 1, -1))*torch.matmul(inv_K[:, :3, :3],self.pix_coords)
        x_s1_item_2  = (depth.view((self.batch_size, 1, -1)) - 1)*self.t_xi         
        x_s1 = x_s1_item_1 - x_s1_item_2 # x_s1(p_t)
        
        x_1 = x_s1 - self.t_xi # x_t(p_t)   
        x_2 = torch.matmul(T[:, :3, :3], x_s1 ) + torch.unsqueeze(T[:, :3, 3], -1) - self.t_xi # x_2(p_t)
        x_1_norm = torch.norm(x_1, 2, 1)
        x_1_norm = torch.unsqueeze(x_1_norm, 1) # x_norm(p_t) 
        x_2_norm = torch.norm(x_2, 2, 1) 
        x_2_norm = torch.unsqueeze(x_2_norm, 1) # x_2_norm(p_t)
        
        
        alpha_1 = (x_1_norm - self.min_depth)/(self.max_depth - self.min_depth) # alpha_1(p_t)
        alpha_2 = (x_2_norm - self.min_depth)/(self.max_depth - self.min_depth) # alpha_2(p_t)
        x_s2_1_min = torch.matmul(T[:, :3, :3], x_1_min) + torch.unsqueeze(T[:, :3, 3], -1) # x_s2_1_min(p_t)
        x_s2_1_max = torch.matmul(T[:, :3, :3], x_1_max) + torch.unsqueeze(T[:, :3, 3], -1) # x_s2_1_max(p_t)
        
        
        #curved_epipolar_constraint = (1 - alpha_1)*x_s2_1_min + alpha_1*x_s2_1_max - ((1-alpha_2)*x_2_min + alpha_2*x_2_max)
        
        
        x_2_dash = torch.matmul(T[:, :3, :3], x_s1) + torch.unsqueeze(T[:, :3, 3], -1) + (x_2_norm - 1)*self.t_xi
        
        
        z_2_dash = torch.unsqueeze(x_2_dash[:, 2, :], 1)
        p_2_tilde = torch.matmul(K[:, :3, :3], x_2_dash) / z_2_dash
        
        
        p_2_tilde[:, :, indexes_outside_circle] = self.pix_coords[:, :, indexes_outside_circle]
         
        pix_coords_converted = p_2_tilde[:, :2, :].view(self.batch_size, 2, self.height, self.width)
        pix_coords_converted = pix_coords_converted.permute(0, 2, 3, 1)
        
        pix_coords_converted_quantized = torch.round(pix_coords_converted)
        pix_coords_indexes_mapping = pix_coords_converted_quantized.reshape((self.batch_size, -1, 2))[:, :, 0] + pix_coords_converted_quantized.reshape((self.batch_size, -1, 2))[:, :, 1]*self.height
        pix_coords_indexes_mapping = pix_coords_indexes_mapping.long()
        
        x_2_max_p_t_dash = self.pixel_index_mapping(x_2_max, pix_coords_indexes_mapping)
        x_2_min_p_t_dash = self.pixel_index_mapping(x_2_min, pix_coords_indexes_mapping)
        
        curved_epipolar_constraint = (1 - alpha_1)*x_s2_1_min + alpha_1*x_s2_1_max - ((1-alpha_2)*x_2_min_p_t_dash + alpha_2*x_2_max_p_t_dash)
        
        pix_coords_converted[..., 0] /= self.width - 1
        pix_coords_converted[..., 1] /= self.height - 1
        pix_coords_converted = (pix_coords_converted - 0.5) * 2

        curved_epipolar_constraint = curved_epipolar_constraint[:, :, indexes_inside_circle]
        
        #print("The shape of 'curved_epipolar_constraint' is {}".format(curved_epipolar_constraint.shape))
        return pix_coords_converted, curved_epipolar_constraint
    
    
def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

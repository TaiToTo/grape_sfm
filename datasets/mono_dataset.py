# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms

import torch.nn as nn



def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    #print("Inside pil_loader() of mono_dataset.py")
    #print(path)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
        
    #print("Outside pil_loader() of mono_dataset.py")
    

def get_solution_area(inv_K, xi, width, height):
    
    #print("Inside get_solution_area()")
    
    temp_batch_size = 1
    
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = nn.Parameter(torch.from_numpy(id_coords), requires_grad=False)
    ones = nn.Parameter(torch.ones(temp_batch_size, 1, height * width), requires_grad=False)
    
    pix_coords = torch.unsqueeze(torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0)
    pix_coords = pix_coords.repeat(temp_batch_size, 1, 1)
    pix_coords = nn.Parameter(torch.cat([pix_coords, ones], 1), requires_grad=False)
    pix_coords = pix_coords[0].transpose(1, 0)
    
    numerator_array = np.zeros(width*height)
    denominator_array = np.zeros(width*height)
    solution_area = np.zeros(width*height).astype(np.int)
    
    for i in range(pix_coords.shape[0]):
        sample_vec = pix_coords[i]
        q_square = np.linalg.norm(np.dot(inv_K, sample_vec)[:2])**2
        #temp = 1 + q_square - xi * q_square
        temp = 1 + q_square - xi**2 * q_square
        
        '''
        if(temp<0):
            numerator_array[i] = 0
            solution_area[i] = -1
        elif(temp==0):
            numerator_array[i] = 0
            solution_area[i] = 0
        else:
            numerator_array[i] = -xi*q_square - np.sqrt(temp)
            solution_area[i] = 1
        '''
        
        if(temp<0):
            numerator_array[i] = 0
            solution_area[i] = 0
        else:
            #numerator_array[i] = -xi*q_square - np.sqrt(temp)
            #numerator_array[i] = -xi*q_square + np.sqrt(temp)
            numerator_array[i] = xi + np.sqrt(temp)
            solution_area[i] = 1

    
    for i in range(pix_coords.shape[0]):
        sample_vec = pix_coords[i]
        q_square = np.linalg.norm(np.dot(inv_K, sample_vec)[:2])**2
        denominator_array[i] = q_square  + 1
        
    numerator_array = numerator_array.reshape(height, width) 
    denominator_array = denominator_array.reshape(height, width)
    solution_area = solution_area.reshape(height, width)
    
    depth_coeffs = np.zeros((height, width))
    depth_coeffs[solution_area!=0] = numerator_array[solution_area!=0] / denominator_array[solution_area!=0]
    #print("Shape of depth_coeffs: {}".format(depth_coeffs.shape))
    #print("Shape of solution_area: {}".format(solution_area.shape))
    #print("Shape of depth_coeffs<0: {}".format((depth_coeffs<0).shape))
    #print("Shape of solution_area[depth_coeffs<0]: {}".format(solution_area[depth_coeffs<0].shape))
    solution_area[depth_coeffs<0] = -1
    #print("Outside get_solution_area()")
    #print()
    solution_area.astype(np.int)
    
    return depth_coeffs, solution_area 
    
    

    
    
    
    
    
class MonoDatasetGrape(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg'):
        super(MonoDatasetGrape, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        
        self.scaled_K_dict = {}
        self.scaled_inv_K_dict = {}
        self.scaled_depth_coeff_dict = {}
        self.scaled_solution_area_dict = {}
        
        self.K = np.array([[427.85318851, 0.0, 241.05120576, 0],
                           [0, 428.12981526, 238.43306185, 0],
                           [0, 0, 1.0, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.xi = 2.0796752
        
        for scale in range(self.num_scales):
            print("Scale checkpoint: {}".format(scale))
            K = self.K.copy()
            
            K[0] = K[0]/2**scale
            K[1] = K[1]/2**scale

            inv_K = np.linalg.pinv(K)
            
            width_rescaled = self.width//2**scale
            height_reslaced = self.height//2**scale
            
            depth_coeffs, solution_area = get_solution_area(inv_K[:3, :3], self.xi, width_rescaled, height_reslaced)
            print("Shape of 'depth_coeffs': {}".format(depth_coeffs.shape))
            
            
            self.scaled_K_dict[scale] = torch.from_numpy(K)
            self.scaled_inv_K_dict[scale] = torch.from_numpy(inv_K)
            self.scaled_depth_coeff_dict[scale] = torch.from_numpy(depth_coeffs.astype(np.float32))
            self.scaled_solution_area_dict[scale] = torch.from_numpy(solution_area)
        

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        #print("Inside __getitem__() of MonoDataset() class")
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        
        #print("self.filenames")
        #print(self.filenames[:20])
        
        line = self.filenames[index].split()
        # 'line' is for example 
        # ['2011_09_26/2011_09_26_drive_0001_sync', '6', 'l']
        #print("'line' is ")
        #print(line)
        folder = line[0]
            
        '''
        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0
        
        if len(line) == 3:
            side = line[2]
        else:
            side = None
        '''    
        
        frame_index = int(line[1])
        #print("Checkpoint A")
        #print("folder: {}".format(folder))
        #print("frame_index: {}".format(frame_index))
        #print("side: {}".format(side))
        #print("do_flip: {}".format(do_flip))
        #print()
        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                #inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, do_flip)
                
                
        # Rescaling instrinsic parameters for each scale.
        for scale in range(self.num_scales):
            #print("Scale checkpoint: {}".format(scale))
            #K = self.K.copy()
            
            #K[0] = K[0]/2**scale
            #K[1] = K[1]/2**scale

            #K[0, :] *= self.width // (2 ** scale)
            #K[1, :] *= self.height // (2 ** scale)

            #inv_K = np.linalg.pinv(K)
            
            #width_rescaled = self.width//2**scale
            #height_reslaced = self.height//2**scale
            
            #solution_coeffs, solution_area = get_solution_area(inv_K[:3, :3], self.xi, width_rescaled, height_reslaced)
            
            inputs[("K", scale)] = self.scaled_K_dict[scale]
            inputs[("inv_K", scale)] = self.scaled_inv_K_dict[scale]
            inputs[("depth_coeff", scale)] = self.scaled_depth_coeff_dict[scale]
            #inputs[("reverse_depth_coeff", scale)] = self.scaled_reverse_depth_coeff_dict[scale]
            inputs[("norm_solution_area", scale)] = self.scaled_solution_area_dict[scale]
            '''
            self.scaled_K_dict[scale] = torch.from_numpy(K)
            self.scaled_inv_K_dict[scale] = torch.from_numpy(inv_K)
            self.scaled_solution_coeffs_dict[scale] = torch.from_numpy(solution_coeffs)
            self.scaled_solution_area_dict[scale] = torch.from_numpy(solution_area)
            '''
            
        #print("Checkpoint C")
        do_color_aug = False
        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        #print("Checkpoint D")
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            #print()
            #print("Checkpoint Depth")
            
            #print("folder, folder_index : ")
            #print(folder)
            #print(frame_index)
            #print()
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)
            
        #print("'inputs.keys()' is ")
        #print(inputs.keys())
        #print("Outisde __getitem__() of MonoDataset() class")
        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
        
       
class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required
        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        
        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            #K = self.K.copy()

            #K[0, :] *= self.width // (2 ** scale)
            #K[1, :] *= self.height // (2 ** scale)

            #inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

        
        

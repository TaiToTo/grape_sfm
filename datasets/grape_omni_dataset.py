# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset, MonoDatasetGrape



'''
        self.K = np.array([[427.85318851, 0.0, 241.05120576, 0],
                           [0, 428.12981526, 238.43306185, 0],
                           [0, 0, 1.0, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.xi = 2.0796752
'''
    
class GrapeDataset(MonoDatasetGrape):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(GrapeDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        

        self.full_res_shape = (480, 480)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index , do_flip):
        #print("Inside get_color() of kitti_dataset.py")
        color = self.loader(self.get_image_path(folder, frame_index))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
            
        #print("Outside get_color() of kitti_dataset.py")

        return color

class GrapeOmniDataset(GrapeDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(GrapeOmniDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index):
        #print("Inside get_image_path() of GrapeStereoProjDataset() class")
        
        f_str = "{}{}".format(frame_index, self.img_ext)
        
        '''
        print("folder: " + str(folder))
        print("self.data_path: " + str(self.data_path))
        print("f_str: " + str(f_str))
        '''
        
        #image_path = os.path.join(
        #    self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        
        image_path = folder + 'frame' + f_str
        #print("folder")
        #print(folder)
        #print("image_path")
        ##print(image_path)
        #print("Outside get_image_path() of GrapeStereoProjDataset() class")
        #print()
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
import glob 
import os
import numpy as np 
import torch
from torch.utils.data import DataLoader

import networks
import datasets
from layers import transformation_from_parameters

pose_model_type = "separate_resnet"
num_pose_frames = 2 

def predict_poses(inputs):
    """Predict poses between input frames for monocular sequences.
    """
    outputs = {}
    if num_pose_frames == 2:
        # In this setting, we compute the pose to each source frame via a
        # separate forward pass through the pose network.

        # select what features the pose network takes as input
        if pose_model_type == "shared":
            #pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            pass
        else:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in frame_ids}

        for f_i in frame_ids[1:]:
            if f_i != "s":
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                if pose_model_type == "separate_resnet":
                    pose_inputs = [pose_encoder(torch.cat(pose_inputs, 1))]
                elif pose_model_type == "posecnn":
                    pose_inputs = torch.cat(pose_inputs, 1)

                axisangle, translation = pose_decoder(pose_inputs)
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

    return outputs


if __name__ == "__main__":
    
    val_data_path = '/workspace/grape_video_compressed_4/YN10101321/R0010293/'
    sample_val_sequence = glob.glob(val_data_path + '*.jpg')
    sample_val_sequence = [(val_data_path + ' ' + str(i+1) ) for i in range(len(sample_val_sequence)-2)]

    data_path = '/workspace/grape_frame_datasets_4'
    height = 480
    width = 480
    num_workers = 12
    batch_size = 1
    frame_ids = [0, -1, 1]
    img_ext = '.jpg'
    dataset = datasets.GrapeOmniDataset
    val_dataset = dataset(
                data_path, sample_val_sequence, height, width,
                frame_ids, 4, is_train=False, img_ext=img_ext)

    val_loader = DataLoader(
                val_dataset, batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True, drop_last=True)
    
    
    model_path = 'models/grape_omni_epipolar_1e-4_modified/models/weights_2'
    #download_model_if_doesnt_exist(model_name)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    encoder = networks.ResnetEncoder(18, False)
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
    depth_decoder.load_state_dict(loaded_dict)

    encoder.cuda()
    depth_decoder.cuda()
    encoder.eval()
    depth_decoder.eval()
    
    
    pose_encoder_path = os.path.join(model_path, "pose_encoder.pth")
    pose_decoder_path = os.path.join(model_path, "pose.pth")

    pose_encoder = networks.ResnetEncoder(18, False, 2)
    pose_encoder.load_state_dict(torch.load(pose_encoder_path))

    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    pose_decoder.load_state_dict(torch.load(pose_decoder_path))

    pose_encoder.cuda()
    pose_encoder.eval()
    pose_decoder.cuda()
    pose_decoder.eval()
    
    
    cam_T_cam_list = []
    with torch.no_grad():
        for idx, inputs in enumerate(val_loader):
            #print(idx)
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            outputs = predict_poses(inputs)
            #print(outputs.keys())
            cam_T_cam_list.append(outputs[('cam_T_cam', 0, 1)][0])

    cam_T_cam_list = [cam_T_cam_list[i].cpu().numpy() for i in range(len(cam_T_cam_list))]
    cam_T_array = np.vstack(cam_T_cam_list).reshape((len(cam_T_cam_list), -1, 4))
    np.save('./cam_pose_array_R0010293.npy', cam_T_array)




import glob
import os
import numpy as np
import torch 
from torch.utils.data import DataLoader


import networks
import datasets


if __name__ == "__main__":
    
    val_data_path = '/workspace/grape_video_compressed_4/YN10101281/R0010110/'
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

    encoder.eval()
    depth_decoder.eval()
    
    
    disp_list = []
    with torch.no_grad():

        for idx, inputs  in enumerate(val_loader):
            input_image_pytorch = inputs[('color', 0, 0)]   
            features = encoder(input_image_pytorch)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            original_width, original_height = 480, 480
            disp_resized = torch.nn.functional.interpolate(disp,
                (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            disp_list.append(disp_resized_np)
    
    
    
    disp_array_to_export = np.vstack(disp_list)
    disp_array_to_export = disp_array_to_export.reshape((len(disp_list),  480, 480))
    np.save('./disp_array_R0010110.npy', disp_array_to_export)

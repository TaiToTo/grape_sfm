# grape_sfm

Code for 3D reconstruction of grape cluster berry positions. It is used in the workflow of unsupervised monocular depth estimation with omnidirectional cameras, berry tracking, and bundle adjustment.

Demo link: https://drive.google.com/drive/folders/1PRcJdTPDC3UI5DEX44y9AAUSXnoQtFDq?usp=sharing

# Unsupervised Monocular Depth Estimation

The code used for Docker environment setup is as follows:

```bash
docker run -v /disk021/usrs/tamura:/workspace \
          --name tamura_monodepth2_env \
          --shm-size 4G --gpus all -itd -p 7775:7775 \
          pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
```

Install necessary libraries as needed. Then, the command to execute training for unsupervised monocular depth estimation in the workspace is as follows:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model_name grape_omni_hikituki_demo \
                                        --log_dir ./models --data_path '/workspace/grape_frame_datasets_4' \
                                        --split 'grape_omni' --dataset 'grape_omni' \
                                        --height 480 \
                                        --width 480 \
                                        --batch_size 8 \
                                        --min_depth 1 \
                                        --max_depth 100 \
                                        --epipolar_weight 1e-4 \
                                        --num_epochs 10
```
Meaning of each argument is as follows:

* model_name: Arbitrarily set name for the model being trained.
* log_dir: Directory where models are saved. Make sure to create it beforehand during training, otherwise the models won't be saved.
* data_path: Directory where the dataset is located.
* split: Name of the training split.
* dataset: Name of the dataset.
* min_depth, max_depth: Range of depth for omnidirectional cameras.
* epipolar_weight: Coefficient of scale_aware_constraint (the original name given, might need to be changed).

The dataset used is located at `/disk021/usrs/tamura/grape_frame_datasets_4` on the server.

Set the path of the trained model to `model_path` and execute `export_camera_pose.py` and `export_disp.py` to output NumPy data necessary for berry tracking (e.g., `cam_pose_array_R0010110.npy`, `disp_array_R0010110.npy`).

```
python export_camera_pose.py
python export_disp.py
```
The implementation is mostly based on Monodepth 2 (https://github.com/nianticlabs/monodepth2).

# Tracking

Using the output camera poses and estimated inverse depths, perform berry tracking between frames using `track_berries.py`. Adjust the data paths appropriately within the `get_initial_data()` function. Set the following parameters:

* local_bunch_frame_interval: Total frames over which bundle adjustment is performed.
* first_frame_idx: Index of the first frame in the consecutive frames for bundle adjustment.

```
python track_berries.py
```

Then, the necessary data for bundle adjustment will be output (e.g., `bundle_data_R0011010_0_49.txt`).

# Bundle Adjustment

Perform bundle adjustment using the tracked berry data with the following code:

```
python main.py
```
The result of bundle adjustment is saved as a pickle file as a class (e.g., `bundle_simultaneous_0_49.pickle`).

The result of bundle adjustment can be visualized using `visualize_partial_bundle_adjustment_results.ipynb`.

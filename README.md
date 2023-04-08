# grape_sfm

ブドウの房の果粒位置の3次元復元のコード．全方位カメラの教師なし単眼深度推定の訓練・推論，果粒のトラッキング，バンドル調整の流れで使用する．


# 教師なし単眼深度推定

Docker の環境構築に使用したコードは以下の通り

```bash
docker run -v /disk021/usrs/tamura:/workspace \
          --name tamura_monodepth2_env \
          --shm-size 4G --gpus all -itd -p 7775:7775 \
          pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
```

適宜必要なライブラリをインストールする．そして，workspace内で教師なし単眼深度推定の訓練を実行するコマンドは以下の通り

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
各引数の意味は以下の通り

* model_name: 訓練するモデルの任意に設定する名前
* log_dir: モデルが保存されるディレクトリ．訓練時にあらかじめ作っておかないとモデルが保存されない. * data_path: データセットをおくディレクトリ
* split: 訓練のスプリットの種類の名前
* dataset: データセットの名前
* min_depth, max_depth: 全方位カメラの深度の範囲
* epipolar_weight: scale_aware_constraint の係数のこと．(最初期につけた名前なので，変更するべき)

訓練したモデルのパスを　model_path に設定して　export_camera_pose.py, export_disp.py　を実行することで，果粒のトラッキングに必要な NumPy データが出力される．(e.g. cam_pose_array_R0010110.npy, disp_array_R0010110.npy)
```
python export_camera_pose.py
python export_disp.py
```
実装はほとんど Monodepth 2 を元にしている. (https://github.com/nianticlabs/monodepth2)

# トラッキング

上記の出力したカメラ姿勢，逆深度の推定結果をもとに track_berries.py でフレーム間の果粒のトラッキングを行う．get_initial_data() 関数の中のデータのパスを適切に設定した上で以下を実行する．

```
python track_berries.py
```

そして，バンドル調整に必要なデータが出力される． (e.g. bundle_data_R0011010_0_49.txt)


# バンドル調整

トラッキングした果粒のデータを元にバンドル調整を以下のコードで行う．

```
python main.py
```
バンドル調整の結果は，クラスとして pickle ファイルで保存される. (e.g. bundle_simultaneous_0_49.pickle)

visualize_partial_bundle_adjustment_results.ipynb　でバンドル調整の結果を可視化することができる．


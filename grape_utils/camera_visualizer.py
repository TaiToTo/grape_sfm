import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


import colorsys
import math

def gradation(position, *, start, stop):
    start_hls = np.array(colorsys.rgb_to_hls(*start))
    stop_hls = np.array(colorsys.rgb_to_hls(*stop))
    delta = stop_hls - start_hls
    hls = start_hls + position * delta
    return colorsys.hls_to_rgb(*hls)

def get_camera_color_list(frame_num, start = (1, 0, 0), stop = (0, 0, 1)):
    color_list = []
    for i in range(frame_num):
        color = gradation(i / max(1, (frame_num - 1)), start=start, stop=stop) 
        color_list.append(color)
    return color_list

def choose_colors(num_colors):
    # matplotlib.colors.CSS4_COLORSの値だけをリストにする
    tmp = list(matplotlib.colors.CSS4_COLORS.values())
    # リストにしたものをランダムにシャッフルする
    random.shuffle(tmp)
    # 必要な数だけ先頭から取り出してreturnする
    label2color = tmp[:num_colors]
    return label2color

class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim, figsize=(18, 7)):
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.gca(projection='3d')
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        #self.ax.set_xlabel('x')
        #self.ax.set_ylabel('y')
        #self.ax.set_zlabel('z')
        print('initialize camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color='r', focal_len_scaled=5, aspect_ratio=0.3, alpha=0.35, exchanged_axes=[0, 1, 2]):
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
        vertex_transformed = np.dot(vertex_std, extrinsic.T)
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]
        #print("Shape of 'meshes[0]': {}".format(meshes[0].shape))
        
        meshes_exchanged = self.exchange_axes_of_meshes(meshes, exchanged_axes)
        self.ax.add_collection3d(
            Poly3DCollection(meshes_exchanged, facecolors=color, linewidths=0.3, edgecolors=color, alpha=alpha))
        

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self):
        plt.title('Extrinsic Parameters')
        plt.show()
        
    def exchange_axes_of_meshes(self, meshes, exchanged_axes = [0, 1, 2]):  
        for mesh_idx in range(len(meshes)):
            for vertex_idx in range(len(meshes[mesh_idx])):
                meshes[mesh_idx][vertex_idx] = meshes[mesh_idx][vertex_idx][exchanged_axes]         
        return meshes
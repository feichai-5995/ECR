#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.


import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['EGL_DEVICE_ID'] = '3'
# os.environ["DISPLAY"] = ":0"
# 对于无图形化界面的服务器，egl肯定是用不了的
# 关键似乎不在于无图形化界面,而在于无权限调用硬件加速
# pyrender渲染不太正常，要不然就换realrender？
os.environ["MUJOCO_GL"] = "osmesa"
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
# 我加的，切换GPU的，看着点


# import pyglet
# pyglet.options['headless'] = True

import argparse
import logging
from distutils.util import strtobool
from pathlib import Path

from trainer import Trainer

# from pyvirtualdisplay import Display



def _strtobool(x):
    return bool(strtobool(x))


if __name__ == '__main__':

    # display = Display(visible=0, size=(800, 600))
    # display.start()

    # Setup logging levels.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Fast training of a scene coordinate regression network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--scene', type=Path, default='datasets/7scenes/7scenes_chess',
                        help='path to a scene in the dataset folder, e.g. "datasets/Cambridge_GreatCourt"')
    parser.add_argument('--output_map_file', type=Path, default='output_final_3/visual/7scenes_chess_visual.pt',
                        help='target file for the trained network')

    parser.add_argument('--encoder_path', type=Path, default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                        help='file containing pre-trained encoder weights')

    parser.add_argument('--num_head_blocks', type=int, default=1,
                        help='depth of the regression head, defines the map size')

    parser.add_argument('--learning_rate_min', type=float, default=0.00025,
                        help='lowest learning rate of 1 cycle scheduler')
    parser.add_argument('--learning_rate_max', type=float, default=0.0075,
                        help='highest learning rate of 1 cycle scheduler')
     
    parser.add_argument('--training_buffer_size', type=int, default=8000000,
                        help='number of patches in the training buffer')

    parser.add_argument('--samples_per_image', type=int, default=1024,
                        help='number of patches drawn from each image when creating the buffer')

    parser.add_argument('--batch_size', type=int, default=5120,
                        help='number of patches for each parameter update (has to be a multiple of 512)')

    parser.add_argument('--epochs', type=int, default=16,
                        help='number of runs through the training buffer')

    parser.add_argument('--repro_loss_hard_clamp', type=int, default=1000,
                        help='hard clamping threshold for the reprojection losses')


    parser.add_argument('--repro_loss_soft_clamp', type=int, default=50,
                        help='soft clamping threshold for the reprojection losses')

    parser.add_argument('--repro_loss_soft_clamp_min', type=int, default=1,
                        help='minimum value of the soft clamping threshold when using a schedule')

    parser.add_argument('--use_half', type=_strtobool, default=True,
                        help='train with half precision')

    parser.add_argument('--use_homogeneous', type=_strtobool, default=True,
                        help='train with half precision')

    parser.add_argument('--use_aug', type=_strtobool, default=True,
                        help='Use any augmentation.')

    parser.add_argument('--aug_rotation', type=int, default=15,
                        help='max inplane rotation angle')

    parser.add_argument('--aug_scale', type=float, default=1.5,
                        help='max scale factor')

    parser.add_argument('--image_resolution', type=int, default=480,
                        help='base image resolution')

    parser.add_argument('--repro_loss_type', type=str, default="dyntanh",
                        choices=["l1", "l1+sqrt", "l1+log", "tanh", "dyntanh"],
                        help='Loss function on the reprojection error. Dyn varies the soft clamping threshold')

    parser.add_argument('--repro_loss_schedule', type=str, default="circle", choices=['circle', 'linear'],
                        help='How to decrease the softclamp threshold during training, circle is slower first')

    parser.add_argument('--depth_min', type=float, default=0.1,
                        help='enforce minimum depth of network predictions')

    parser.add_argument('--depth_target', type=float, default=10,
                        help='default depth to regularize training')

    parser.add_argument('--depth_max', type=float, default=100,
                        help='enforce maximum depth of network predictions')

    # Clustering params, for the ensemble training used in the Cambridge experiments. Disabled by default.
    parser.add_argument('--num_clusters', type=int, default=None,
                        help='split the training sequence in this number of clusters. disabled by default')

    parser.add_argument('--cluster_idx', type=int, default=None,
                        help='train on images part of this cluster. required only if --num_clusters is set.')
    # Params for the visualization. If enabled, it will slow down training considerably. But you get a nice video :)
    parser.add_argument('--render_visualization', type=_strtobool, default=True,
                        help='create a video of the mapping process')

    parser.add_argument('--render_target_path', type=Path, default='output_final_3/visual',
                        help='target folder for renderings, visualizer will create a subfolder with the map name')

    parser.add_argument('--render_flipped_portrait', type=_strtobool, default=False,
                        help='flag for wayspots dataset where images are sideways portrait')

    parser.add_argument('--render_map_error_threshold', type=int, default=10,
                        help='reprojection error threshold for the visualisation in px')

    parser.add_argument('--render_map_depth_filter', type=int, default=10,
                        help='to clean up the ACE point cloud remove points too far away')

    parser.add_argument('--render_camera_z_offset', type=int, default=4,
                        help='zoom out of the scene by moving render camera backwards, in meters')

    options = parser.parse_args()

    trainer = Trainer(options)
    trainer.train()

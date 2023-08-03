import torch
import os
import os.path as osp
import numpy as np
import struct
import open3d
import math
import imageio
import cv2
import pdb
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from pypcd import pypcd
import json


def save_lidar_data(lidar_data, export_all_points=True, save_dir=""):
    coords = np.ascontiguousarray(lidar_data[:, :3])
    category_id = np.ascontiguousarray(lidar_data[:, -1]).astype(int)

    category_id[category_id > 5] = 0
    torch.save((coords, 0, category_id), save_dir)


def convert_autra_to_openscene_train_format(autra_train_data, openscene_train_data):
    for scene_index in tqdm(os.listdir(autra_train_data)):
        lidar_file = osp.join(autra_train_data, scene_index)
        scene_name = scene_index.split("-lidar")[0]
        # save lidar
        try:
            pcd = pypcd.PointCloud.from_path(lidar_file).pc_data
        except:
            print("error for ", lidar_file)

        coors = np.concatenate([
            pcd['x'], pcd['y'], pcd['z'], pcd['intensity'],
            pcd['rgb'], pcd['label']]).astype(np.float32).reshape(6, -1).T

        valid_point_mask = ~((coors[:, 0] == 0) & (coors[:, 1] == 0) & (coors[:, 2] == 0))
        coors = coors[valid_point_mask]
        save_dir = osp.join(openscene_train_data, 'nuscenes_autra_3d_detect_dem', 'train')
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        save_file = osp.join(openscene_train_data, 'nuscenes_autra_3d_detect_dem', 'train', scene_name + ".pth")
        save_lidar_data(coors, True, save_file)

def main():
    autra_train_data = "/mnt/cfs/agi/workspace/LidarAnnotation/data/Robin_20230517_164224/lidar/"
    openscene_train_data = "data/"
    convert_autra_to_openscene_train_format(autra_train_data, openscene_train_data)

if __name__ == '__main__':
    main()

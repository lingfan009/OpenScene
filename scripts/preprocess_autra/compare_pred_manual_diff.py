import torch
import os
import os.path as osp
import numpy as np
import struct
import open3d
import math
import imageio
import argparse
from tqdm import tqdm
import cv2
from scipy.spatial.transform import Rotation as R
from pypcd import pypcd
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import torch

def compare_pred_manual_diff():
    #args = parse_args()
    # 1685598346022-Robin-lidar.pcd
    pred_label_path = "/mnt/cfs/agi/workspace/LidarAnnotation/data/nuscenes_autra_3d_dataset_v4_pcd/lidar"
    # 1685608315021-ACRush.pth
    manual_label_path = "/home/fan.ling/big_model/OpenScene/OpenScene/data/point_cloud_label_3d/nuscenes_autra_3d_dataset_v4/autra_seg_eval_v1.01/train"
    manual_image_path = "/mnt/cfs/agi/data/pretrain/sun/provider_manual_seg_result/2023060202_180000-longmao-uniform_sample-discrete-seg_only"
    save_lidar_diff_path = "/mnt/cfs/agi/workspace/LidarAnnotation/data/nuscenes_autra_3d_dataset_v4_pred_manual_diff_debug/lidar"
    save_image_path = "/mnt/cfs/agi/workspace/LidarAnnotation/data/nuscenes_autra_3d_dataset_v4_pred_manual_diff_debug/camera"

    for file_name in tqdm(os.listdir(manual_label_path)):
        manual_label_file = osp.join(manual_label_path, file_name)
        manual_pred_file = osp.join(pred_label_path, file_name.replace(".pth", "-lidar.pcd"))
        frame_name = file_name.split(".")[0]
        # load manual point
        try:
            #pcd_manual = pypcd.PointCloud.from_path(manual_label_file).pc_data
            #pcd_manual = np.fromfile(manual_label_file, dtype=np.float32)
            #data_np = pcd.reshape(-1, 5)
            locs_in, feats_in, labels_in = torch.load(manual_label_file)

        except:
            print("error for ", manual_label_file)
            continue
        # coors_manual = np.concatenate([
        #     pcd_manual['x'], pcd_manual['y'], pcd_manual['z'], pcd_manual['intensity'],
        #     pcd_manual['rgb'], pcd_manual['label']]).astype(np.float32).reshape(6, -1).T
        coors_manual =labels_in

        # load pred point
        try:
            pcd_pred = pypcd.PointCloud.from_path(manual_pred_file)
        except:
            print("error for ", manual_pred_file)
            continue

        coors_pred = np.concatenate([
            pcd_pred.pc_data['x'], pcd_pred.pc_data['y'], pcd_pred.pc_data['z'],
            pcd_pred.pc_data['rgb'], pcd_pred.pc_data['label']]).astype(np.float32).reshape(5, -1).T

        assert coors_manual.shape[0] == coors_pred.shape[0], "pred and manual mismatch error"

        point_match = coors_manual[:] == coors_pred[:, -1]
        pcd_pred.pc_data['label'][point_match] = 999
        pcd_pred.pc_data['rgb'][point_match] = 14474460

        # save lidar point
        if not osp.exists(save_lidar_diff_path):
            os.makedirs(save_lidar_diff_path)
        lidar_save_file = osp.join(save_lidar_diff_path, frame_name + '.pcd')
        pcd_pred.save_pcd(lidar_save_file, compression='binary')

        for camera_file in os.listdir(osp.join(manual_image_path, frame_name, "image")):
            camera_type = camera_file.split("-")[-1].split(".")[0]
            image_save_dir = osp.join(save_image_path, camera_type)
            if not osp.exists(image_save_dir):
                os.makedirs(image_save_dir)
            source_image_file = osp.join(manual_image_path, frame_name, "image", camera_file)
            image_save_file = osp.join(image_save_dir, frame_name+".jpg")
            os.system(f"cp {source_image_file} {image_save_file}")

if __name__ == '__main__':

    compare_pred_manual_diff()
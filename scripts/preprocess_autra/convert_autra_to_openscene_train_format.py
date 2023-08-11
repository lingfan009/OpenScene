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

cam_types = ['camera_upmiddle_right', 'camera_upmiddle_middle', 'camera_upmiddle_left', 'camera_left_front', 'camera_left_backward', 'camera_right_front', 'camera_right_backward']
cam_type_dict = {
    'camera_upmiddle_right': 'upmiddle_right_60h',
    'camera_upmiddle_middle': 'upmiddle_middle_120h',
    'camera_upmiddle_left': 'upmiddle_left_30h',
    'camera_left_front': 'left_front_120h',
    'camera_left_backward': 'left_backward_120h',
    'camera_right_front': 'right_front_120h',
    'camera_right_backward': 'right_backward_120h'
}

def get_clip_text_embedding():
    text_embedding_path = "saved_text_embeddings/text_embedding_feature_6_cls.pth"
    text_embedding_feature = torch.load(text_embedding_path)["text_embedding_feature"].numpy()
    return text_embedding_feature

def save_lidar_data(lidar_data, export_all_points=True, save_dir=""):
    coords = np.ascontiguousarray(lidar_data[:, :3])
    category_id = np.ascontiguousarray(lidar_data[:, -1]).astype(int)

    category_id[category_id > 5] = 0
    torch.save((coords, 0, category_id), save_dir)

def convert_autra_to_openscene_train_format(label_path, openscene_train_data, openscene_feature_data, package_root_path, dataset_type, dataset_name):
    record_name = label_path.split("/")[-1]
    package_path = osp.join(package_root_path, record_name)

    for index_name in tqdm(os.listdir(package_path)):
        frame_name = os.listdir(osp.join(package_path, index_name))[0]
        # save camera
        camera_type_str = "camera_upmiddle_right;camera_upmiddle_middle;camera_upmiddle_left;camera_left_front;camera_left_backward;camera_right_front;camera_right_backward"
        camera_type_list = camera_type_str.split(";")
        target_camera_root = f"/mnt/cfs/agi/workspace/LidarAnnotation/data/{record_name}/camera"

        for camera_type in camera_type_list:
            source_camera_file = osp.join(package_path, index_name, frame_name, camera_type, frame_name + "-" + camera_type + ".jpg")
            target_camera_type_root = osp.join(target_camera_root, camera_type)
            if not osp.exists(target_camera_type_root):
                os.makedirs(target_camera_type_root, exist_ok=True)
            target_camera_file = osp.join(target_camera_type_root, frame_name + "-lidar.jpg")
            if (not osp.exists(target_camera_file)) and osp.exists(source_camera_file):
                os.system(f"cp {source_camera_file} {target_camera_file}")

        lidar_file = osp.join(label_path, "detect_dem_seg_label", frame_name + "-lidar.pcd")
        if not osp.exists(lidar_file):
            continue
        # save lidar point and label
        try:
            pcd = pypcd.PointCloud.from_path(lidar_file).pc_data
        except:
            print("error for ", lidar_file)
            continue

        coors = np.concatenate([
            pcd['x'], pcd['y'], pcd['z'], pcd['intensity'],
            pcd['rgb'], pcd['label']]).astype(np.float32).reshape(6, -1).T

        valid_point_mask = ~((coors[:, 0] == 0) & (coors[:, 1] == 0) & (coors[:, 2] == 0))
        coors = coors[valid_point_mask]
        lidar_save_dir = osp.join(openscene_train_data, dataset_name, dataset_type)
        if not osp.exists(lidar_save_dir):
            os.makedirs(lidar_save_dir)
        lidar_save_file = osp.join(lidar_save_dir, frame_name + ".pth")
        if osp.exists(lidar_save_file):
            continue
        save_lidar_data(coors, True, lidar_save_file)

        # save label feature & fill text embedding features
        text_embeddings = get_clip_text_embedding()
        points_with_feature = np.zeros((coors.shape[0], 768))
        labels = np.ascontiguousarray(coors[:, -1]).astype(int)
        labels[labels > 5] = 0
        points_with_feature[:] = text_embeddings[labels]
        mask_entire = labels <= 5

        # save point feature
        points_with_feature_torch = torch.from_numpy(points_with_feature)
        feature_save_dir = osp.join(openscene_feature_data, dataset_name)
        if not osp.exists(feature_save_dir):
            os.makedirs(feature_save_dir, exist_ok=True)
        feature_save_file = osp.join(feature_save_dir, frame_name + ".pth")
        torch.save({"feat": points_with_feature_torch.half().cpu(), "mask_full": mask_entire}, feature_save_file)

def parse_args():
    args = argparse.ArgumentParser(description="Data processor for data labeling pipeline.")
    args.add_argument('--record_name', type=str, default='', help='config file path')
    return args.parse_args()

def main():
    args = parse_args()

    openscene_train_point_data = "data/point_cloud_label_3d"
    openscene_train_feature_data = "data/text_feature_3d/"
    dataset_type = "train"
    dataset_name = "nuscenes_autra_3d_dataset_v1"
    record_path = f"/mnt/cfs/agi/data/pretrain/sun/auto_label_data/{args.record_name}"
    package_root_path = "/mnt/cfs/agi/data/pretrain/sun/package_data"
    convert_autra_to_openscene_train_format(record_path, openscene_train_point_data, openscene_train_feature_data, package_root_path, dataset_type, dataset_name)

if __name__ == '__main__':
    main()

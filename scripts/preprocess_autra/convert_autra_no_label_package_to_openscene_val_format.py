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

# def get_clip_text_embedding():
#     text_embedding_path = "/home/fan.ling/big_model/OpenScene/OpenScene/fuse_2d_features/nuscenes_autra_2d_test/text_embedding_feature_6_cls.pth"
#     text_embedding_feature = torch.load(text_embedding_path)["text_embedding_feature"].numpy()
#     return text_embedding_feature

def get_clip_text_embedding():
    text_embedding_path = "model_checkpoint/saved_text_embeddings/text_embedding_feature_6_cls.pth"
    text_embedding_feature = torch.load(text_embedding_path)["text_embedding_feature"].numpy()
    default_embed = np.zeros((1000,768))
    text_embedding_feature = np.concatenate([text_embedding_feature, default_embed], axis=0)
    return text_embedding_feature


def convert_autra_to_openscene_train_format(autra_train_data, openscene_train_data, dataset_name):
    for index_name in tqdm(os.listdir(autra_train_data)):
        frame_name = os.listdir(osp.join(autra_train_data, index_name))[0]

        # save camera
        camera_type_str = "camera_upmiddle_right;camera_upmiddle_middle;camera_upmiddle_left;camera_left_front;camera_left_backward;camera_right_front;camera_right_backward"
        camera_type_list = camera_type_str.split(";")
        target_camera_root = "/mnt/cfs/agi/workspace/LidarAnnotation/data/2023090401_180000-longmao-data_mining-discrete-lidar_only/camera"
        for camera_type in camera_type_list:
            source_camera_file = osp.join(autra_train_data, index_name, frame_name, camera_type, frame_name + "-" + camera_type + ".jpg")
            target_camera_type_root = osp.join(target_camera_root, camera_type)
            if not osp.exists(target_camera_type_root):
                os.makedirs(target_camera_type_root)
            target_camera_file = osp.join(target_camera_type_root, frame_name + "-lidar.jpg")
            os.system(f"cp {source_camera_file} {target_camera_file}")

        lidar_file = osp.join(autra_train_data, index_name, frame_name, "lidar", frame_name + "-lidar.pcd")
        # save lidar
        try:
            pcd = pypcd.PointCloud.from_path(lidar_file).pc_data
        except:
            print("error for ", lidar_file)
            continue

        data_np = np.concatenate([
            pcd['x'], pcd['y'], pcd['z'], pcd['intensity'],
            pcd['timestamp']
        ]).astype(np.float32).reshape(5, -1).T

        valid_point_mask = ~((data_np[:, 0] == 0) & (data_np[:, 1] == 0) & (data_np[:, 2] == 0))
        data_np = data_np[valid_point_mask]
        
        save_dir = osp.join(openscene_train_data, "point_cloud_label_3d", "nuscenes_autra_3d_dataset_v4", dataset_name, "train")
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        save_file = osp.join(save_dir, frame_name + "-lidar.pth")

        coords = np.ascontiguousarray(data_np[:, :3])
        intensity = np.ascontiguousarray(data_np[:, 3:4]).astype(int)
        category_id = np.ascontiguousarray(data_np[:, -1]).astype(int)
        category_id[:] = 0
        torch.save((coords, intensity, category_id), save_file)

        # coors = np.ascontiguousarray(data_np[:, :3])
        # category_id = np.ascontiguousarray(data_np[:, -2]).astype(int)
        # category_id[:] = 0
        # torch.save((coors, 0, category_id), save_file)
        # #save_lidar_data(coors, True, save_file)

        # save label feature
        # # fill text embedding features
        text_embeddings = get_clip_text_embedding()
        points_with_feature = np.zeros((coords.shape[0], 768))
        labels = category_id
        points_with_feature[:] = text_embeddings[labels]
        mask_entire = labels <= 5

        points_with_feature_torch = torch.from_numpy(points_with_feature)
        save_dir = osp.join(openscene_train_data, "text_feature_3d", "nuscenes_autra_3d_dataset_v4", dataset_name)
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        save_file = osp.join(save_dir, frame_name + "-lidar.pth")
        torch.save({"feat": points_with_feature_torch.half().cpu(), "mask_full": mask_entire}, save_file)

def main():
    autra_train_data = "/data03/lingfan/data_labeling/dataset/longmao_labeling/2023090401/package_data/2023090401_180000-longmao-data_mining-discrete-lidar_only"
    openscene_train_data = "data/"
    for case_name in os.listdir(autra_train_data):
        autra_case_train_data = osp.join(autra_train_data, case_name)
        if os.listdir(osp.join(autra_train_data, case_name))[0] == "000001":
            autra_case_train_data = osp.join(autra_train_data, case_name, "000001")
        convert_autra_to_openscene_train_format(autra_case_train_data, openscene_train_data, "2023090401_180000-longmao-data_mining-discrete-lidar_only")

if __name__ == '__main__':
    main()

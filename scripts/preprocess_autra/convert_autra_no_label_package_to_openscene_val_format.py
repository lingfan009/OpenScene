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


cate_map = np.zeros((25,))
cate_map[2] = 5
cate_map[3] = 5
cate_map[4] = 5
cate_map[7] = 5

cate_map[8] = 4

cate_map[11] = 2

cate_map[12] = 3
cate_map[13] = 3
cate_map[14] = 3
cate_map[15] = 3
cate_map[16] = 3
cate_map[17] = 3
cate_map[18] = 3

cate_map[19] = 1
cate_map[20] = 1
cate_map[21] = 1
cate_map[22] = 1
cate_map[23] = 1

print(cate_map)
def get_clip_text_embedding():
    text_embedding_path = "/home/fan.ling/big_model/OpenScene/OpenScene/fuse_2d_features/nuscenes_autra_2d_test/text_embedding_feature_6_cls.pth"
    text_embedding_feature = torch.load(text_embedding_path)["text_embedding_feature"].numpy()
    return text_embedding_feature


def convert_autra_to_openscene_train_format(autra_train_data, openscene_train_data):
    for index_name in tqdm(os.listdir(autra_train_data)):
        frame_name = os.listdir(osp.join(autra_train_data, index_name))[0]

        # save camera
        camera_type_str = "camera_upmiddle_right;camera_upmiddle_middle;camera_upmiddle_left;camera_left_front;camera_left_backward;camera_right_front;camera_right_backward"
        camera_type_list = camera_type_str.split(";")
        target_camera_root = "/mnt/cfs/agi/workspace/LidarAnnotation/data/nuscenes_autra_3d_no_label_package/camera"
        for camera_type in camera_type_list:
            source_camera_file = osp.join(autra_train_data, index_name, frame_name, camera_type, frame_name + "-" + camera_type + ".jpg")
            target_camera_type_root = osp.join(target_camera_root, camera_type)
            if not osp.exists(target_camera_type_root):
                os.makedirs(target_camera_type_root)
            target_camera_file = osp.join(target_camera_type_root, frame_name + "-lidar.jpg")
            os.system(f"cp {source_camera_file} {target_camera_file}")

        continue
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


        print(data_np.shape)
        valid_point_mask = ~((data_np[:, 0] == 0) & (data_np[:, 1] == 0) & (data_np[:, 2] == 0))
        data_np = data_np[valid_point_mask]
        print("after filter:", data_np.shape)
        
        save_dir = osp.join(openscene_train_data, 'nuscenes_autra_3d_no_label_package', 'val')
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        save_file = osp.join(openscene_train_data, 'nuscenes_autra_3d_no_label_package', 'val', frame_name + ".pth")

        coors = np.ascontiguousarray(data_np[:, :3])
        category_id = np.ascontiguousarray(data_np[:, -2]).astype(int)
        #category_id[:] = cate_map[category_id]
        category_id[:] = 0
        print(np.unique(category_id, return_counts=True))
        torch.save((coors, 0, category_id), save_file)
        #save_lidar_data(coors, True, save_file)

        # save label feature
        # # fill text embedding features
        text_embeddings = get_clip_text_embedding()
        points_with_feature = np.zeros((coors.shape[0], 768))
        labels = category_id
        points_with_feature[:] = text_embeddings[labels]
        mask_entire = labels <= 5

        #pdb.set_trace()
        # save point feature
        points_with_feature_torch = torch.from_numpy(points_with_feature)

        save_dir = "/home/fan.ling/big_model/OpenScene/OpenScene/fuse_2d_features/nuscenes_autra_2d_test/nuscenes_autra_3d_no_label_package/"
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        save_file = osp.join(save_dir,frame_name + ".pth")
        torch.save({"feat": points_with_feature_torch.half().cpu(), "mask_full": mask_entire}, save_file)

def main():
    autra_train_data = "/home/fan.ling/big_model/OpenScene/OpenScene/data/val_data_with_no_label/2023072101_180000-longmao-case_mining-continuous-lidar_only"
    openscene_train_data = "data/"
    for case_name in os.listdir(autra_train_data):
        autra_case_train_data = osp.join(autra_train_data, case_name)
        if os.listdir(osp.join(autra_train_data, case_name))[0] == "000001":
            autra_case_train_data = osp.join(autra_train_data, case_name, "000001")
        convert_autra_to_openscene_train_format(autra_case_train_data, openscene_train_data)

if __name__ == '__main__':
    main()

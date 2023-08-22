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

dataset_version_map = dict()
dataset_version_map['2023060202_180000-longmao-uniform_sample-discrete-seg_only'] = "autra_seg_eval_v1.01"

def get_clip_text_embedding():
    text_embedding_path = "/home/fan.ling/big_model/OpenScene/OpenScene/model_checkpoint/saved_text_embeddings/text_embedding_feature_6_cls.pth"
    text_embedding_feature = torch.load(text_embedding_path)["text_embedding_feature"].numpy()
    return text_embedding_feature


def convert_autra_to_openscene_train_format(autra_eval_data, openscene_eval_data):
    batch_name = autra_eval_data.split("/")[-1]
    dataset_version_name = dataset_version_map[batch_name]

    for scene_name in tqdm(os.listdir(autra_eval_data)):
        label_file = osp.join(autra_eval_data, scene_name, "label_result", scene_name+".bin")
        # save lidar
        try:
            pcd = np.fromfile(label_file, dtype=np.float32)
            data_np = pcd.reshape(-1, 5)
        except:
            print("error for ", label_file)

        valid_point_mask = ~((data_np[:, 0] == 0) & (data_np[:, 1] == 0) & (data_np[:, 2] == 0))
        data_np = data_np[valid_point_mask]

        data_np_copy = np.copy(data_np)
        data_np[:, 0] = -data_np_copy[:, 1]
        data_np[:, 1] = data_np_copy[:, 0]
        
        save_dir = osp.join(openscene_eval_data, 'point_cloud_label_3d', 'nuscenes_autra_3d_dataset_v3', dataset_version_name, 'train')
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        save_file = osp.join(save_dir, scene_name + ".pth")

        coors = np.ascontiguousarray(data_np[:, :3])
        intensity = np.ascontiguousarray(data_np[:, 3:4]).astype(int)
        category_id = np.ascontiguousarray(data_np[:, -1]).astype(int)
        category_id[:] = cate_map[category_id]
        print(np.unique(category_id, return_counts=True))
        print(np.unique(intensity, return_counts=True))
        torch.save((coors, intensity, category_id), save_file)
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

        save_dir = osp.join(openscene_eval_data, 'text_feature_3d', 'nuscenes_autra_3d_dataset_v3', dataset_version_name)
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        save_file = osp.join(save_dir, scene_name + ".pth")
        torch.save({"feat": points_with_feature_torch.half().cpu(), "mask_full": mask_entire}, save_file)

def main():
    autra_eval_data = "/mnt/cfs/agi/data/pretrain/sun/provider_manual_seg_result/2023060202_180000-longmao-uniform_sample-discrete-seg_only"
    openscene_eval_data = "data/"
    convert_autra_to_openscene_train_format(autra_eval_data, openscene_eval_data)

if __name__ == '__main__':
    main()

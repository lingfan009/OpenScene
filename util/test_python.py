import os

import torch
import numpy as np
import os.path as osp

NUSCENES16_COLORMAP = {
    1: (220,220,  0), # barrier
    2: (119, 11, 32), # bicycle
    3: (0, 60, 100), # bus
    4: (0, 0, 250), # car
    5: (230,230,250), # construction vehicle
    6: (0, 0, 230), # motorcycle
    7: (220, 20, 60), # person
    8: (250, 170, 30), # traffic cone
    9: (200, 150, 0), # trailer
    10: (0, 0, 110) , # truck
    11: (128, 64, 128), # road
    12: (0,250, 250), # other flat
    13: (244, 35, 232), # sidewalk
    14: (152, 251, 152), # terrain
    15: (70, 70, 70), # manmade
    16: (107,142, 35), # vegetation
    17: (0, 0, 0), # unknown
    }

def test_color_type():
    color_map = {}
    for k,v in NUSCENES16_COLORMAP.items():
        print(k)
        print(v[1])
        color_type = v[0] + v[1] + v[2]
        color_map[color_type] = k

    print(len(color_map))
    print(color_map)

    point_color_type_map = np.zeros(1000)
    for k,v in color_map.items():
        point_color_type_map[k] = v
    print(point_color_type_map)

    point_color = np.array([160.0,160,440]).astype(np.int)
    print(point_color)


    point_color_type = point_color_type_map[point_color]
    print(point_color_type.reshape(-1,1))

def update_annos():
    train_data_version = [
        'autra_train_v2.01/annos_rm_mc.pkl',
        'autra_train_v2.02/annos_rm_mc.pkl',
        'autra_train_v2.03/annos_rm_mc.pkl',
        'autra_train_v2.04/annos_rm_mc.pkl',
        'autra_train_v2.05/annos_rm_mc.pkl',
        'autra_train_v2.06/annos_rm_mc.pkl',
        'autra_train_v2.07/annos_rm_mc.pkl',
        'autra_train_v2.08/annos_rm_mc.pkl',
        'autra_train_v2.09/annos_rm_mc.pkl',
        'autra_train_v2.10/annos_rm_mc.pkl',
        'autra_train_v2.11/annos_rm_mc.pkl',
        'autra_train_v2.12/annos_rm_mc.pkl',
        'autra_train_v2.13/annos_V1.pkl',
        'autra_train_v2.14/annos_V1.pkl',
        'autra_train_v2.15/annos_V1.pkl',
        'autra_train_v2.16/annos_V2.pkl',
        'autra_train_v2.17/annos_rm_mc.pkl',
        'autra_train_v2.18/annos_V2.pkl',
        'autra_train_v2.19/annos_V2.pkl',
        'autra_train_v2.20/annos_V2.pkl',
        'autra_train_v2.21/annos_V1.pkl',
        'autra_train_v2.22/annos_V1.pkl',
        'autra_train_v2.23/annos_V1.pkl',
        'autra_train_v2.24/annos.pkl',
        'autra_train_v2.25/annos.pkl',
        'autra_train_v2.27/annos.pkl',
        'autra_train_v2.28/annos.pkl',
        'autra_train_v2.29/annos.pkl',
        'autra_eval_v2.01/annos_rm_mc.pkl',
        'autra_eval_v2.02/annos_rm_mc.pkl',
        'cangzhou_shuju/annos_V1.pkl',
        'autra_train_case_v1.0/annos.pkl',
        'autra_train_case_v1.1/annos.pkl',
        'autra_train_case_v1.2/annos.pkl',
        'autra_train_case_v1.3/annos.pkl',
        'autra_train_case_v1.4/annos.pkl',
        'autra_union_lidar_dataset_v1.01/annos_lidar_type.pkl',
        'autra_union_lidar_dataset_v1.02/annos_lidar_type.pkl',
        'autra_union_lidar_dataset_v1.03/annos_lidar_type.pkl',
        'autra_union_lidar_dataset_v1.04/annos_lidar_type.pkl',
        'autra_eval_v2.05/ACRUSH/annos.pkl',
        'autra_eval_v2.05/JAMES/annos_JMS.pkl',
        'autra_eval_v2.05/ROBIN/annos_RBN.pkl',
        'autra_eval_v2.06/annos.pkl'
    ]

    perception_train_data_root = "tos:perception-dataset-v2"
    perception_train_data_save_root = "tos:perception-dataset-v2/pre_labeled_dataset/self_learning"
    #perception_train_data_save_root = "tos:data-labeling-huoshan/self_learning"

    for version in train_data_version:
        if "autra_union_lidar_dataset_v1.04" in version or "autra_eval_v2.05" in version or True:
            data_version = version.split("annos")[0]
            anno_name = version.split("/")[-1]
            train_data_path = osp.join(perception_train_data_root, version)
            save_data_path = osp.join(perception_train_data_save_root, data_version)
            local_annos = osp.join("./test_data/", anno_name)
            os.system(f"rclone copy {train_data_path} {local_annos}")
            os.system(f"rclone copy {local_annos} {save_data_path}")
            os.system(f"rm -rf {local_annos}")



if __name__ == "__main__":
    #test_color_type()
    update_annos()
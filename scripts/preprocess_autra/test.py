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

def download_perception_train():
    dataset_version_list = [   
        "autra_eval_m1.1",
        "autra_eval_m1.2",
        "autra_eval_rain_v1.0",
        "autra_eval_v2.01",
        "autra_eval_v2.02",
        "autra_eval_v2.03",
        "autra_eval_v2.04",
        "autra_eval_v2.05",
        "autra_eval_v2.06",
        "autra_eval_v2.07",
        "autra_eval_v2.08",
        "autra_eval_v2.09",
        "autra_train_case_m1.1",
        "autra_train_case_m1.2",
        "autra_train_case_v1.0",
        "autra_train_case_v1.1",
        "autra_train_case_v1.10",
        "autra_train_case_v1.11",
        "autra_train_case_v1.12",
        "autra_train_case_v1.13",
        "autra_train_case_v1.14",
        "autra_train_case_v1.15",
        "autra_train_case_v1.16",
        "autra_train_case_v1.17",
        "autra_train_case_v1.2",
        "autra_train_case_v1.3",
        "autra_train_case_v1.4",
        "autra_train_case_v1.5",
        "autra_train_case_v1.6",
        "autra_train_case_v1.7",
        "autra_train_case_v1.8",
        "autra_train_case_v1.9",
        "autra_train_m0.1",
        "autra_train_m0.2",
        "autra_train_m0.3",
        "autra_train_m0.3_fix",
        "autra_train_m0.4",
        "autra_train_m0.5",
        "autra_train_m1.1",
        "autra_train_m1.2",
        "autra_train_m1.3",
        "autra_train_m1.4",
        "autra_train_m1.5",
        "autra_train_m1.6",
        "autra_train_m1.7",
        "autra_train_m1.8",
        "autra_train_v2.01",
        "autra_train_v2.02",
        "autra_train_v2.03",
        "autra_train_v2.04",
        "autra_train_v2.05",
        "autra_train_v2.06",
        "autra_train_v2.07",
        "autra_train_v2.08",
        "autra_train_v2.09",
        "autra_train_v2.10",
        "autra_train_v2.11",
        "autra_train_v2.12",
        "autra_train_v2.13",
        "autra_train_v2.14",
        "autra_train_v2.15",
        "autra_train_v2.16",
        "autra_train_v2.17",
        "autra_train_v2.18",
        "autra_train_v2.19",
        "autra_train_v2.20",
        "autra_train_v2.21",
        "autra_train_v2.22",
        "autra_train_v2.23",
        "autra_train_v2.24",
        "autra_train_v2.25",
        "autra_train_v2.27",
        "autra_train_v2.28",
        "autra_train_v2.29",
        "autra_train_v2.30",
        "autra_train_v2.31",
        "autra_train_v2.32",
        "autra_train_v2.33",
        "autra_train_v2.34",
        "autra_train_v2.35",
        "autra_train_v2.36",
        "autra_union_lidar_dataset_v1.01",
        "autra_union_lidar_dataset_v1.02",
        "autra_union_lidar_dataset_v1.03",
        "autra_union_lidar_dataset_v1.04"
    ]
    for dataset_version in dataset_version_list:
        os.system(f"rclone copy  tos:perception-dataset-v2/{dataset_version}/bins  /mnt/cfs/agi/lingfan/perception-dataset-v2/{dataset_version}/bins --update --fast-list --timeout=0 --transfers=48 --progress --s3-disable-checksum ")
        os.system(f"rclone copy  tos:perception-dataset-v2/{dataset_version}/annos.pkl  /mnt/cfs/agi/lingfan/perception-dataset-v2/{dataset_version}/ --update --fast-list --timeout=0 --transfers=48 --progress --s3-disable-checksum ")


def process_one_scene1(params):
    try:
        #print(params)
        processed_data = torch.load(params[0])
        feat_3d, mask_chunk = processed_data['feat'], processed_data['mask_full']
    except:
        print(params[0])

def main():

    openscene_train_point_data = "data/point_cloud_label_3d/nuscenes_autra_3d_dataset_v1/train/"
    openscene_train_feature_data = "data/text_feature_3d/nuscenes_autra_3d_dataset_v1"

    print("enter")
    total_num = len(os.listdir(openscene_train_point_data))
    index = 0
    data_paths_list = []
    
    with ThreadPoolExecutor(max_workers=20) as pool:
        for pcd_name in tqdm( os.listdir(openscene_train_point_data)):
            pcd_file = osp.join(openscene_train_point_data, pcd_name)
            feature_file = osp.join(openscene_train_feature_data, pcd_name)
            
            if not osp.exists(feature_file):
                print(pcd_file)
                #os.system(f"rm {pcd_file}")
            feature_file = osp.join(openscene_train_feature_data, pcd_name[:-4].split('/')[-1] + '.pth')
            index += 1
            
            # 使用线程执行map计算
            # 后面元组有3个元素，因此程序启动3次线程来执行action函数
            data_paths_list.append((feature_file,))
            if len(data_paths_list) == 15 or index == total_num:
                results = pool.map(process_one_scene1, data_paths_list)
                print('--------------')
                for r in results:
                    pass
                data_paths_list.clear()



if __name__ == '__main__':
    main()

    #download_perception_train()

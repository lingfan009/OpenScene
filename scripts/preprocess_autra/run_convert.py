import torch
import os
import os.path as osp
import numpy as np
import struct
import time
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

def get_process_order(root_dir):
    dataset_list = []
    for ele in os.listdir(root_dir):
        if ele.startswith("autra_train_v"):
            dataset_list.append(ele)
    for ele in os.listdir(root_dir):
        if ele.startswith("autra_eval_v"):
            dataset_list.append(ele)
    for ele in os.listdir(root_dir):
        if ele.startswith("autra_train_case_v"):
            dataset_list.append(ele)
    for ele in os.listdir(root_dir):
        if ele.startswith("autra_train_m"):
            dataset_list.append(ele)
    for ele in os.listdir(root_dir):
        if ele not in dataset_list and ele.startswith("autra"):
            dataset_list.append(ele)
    print(dataset_list)
    return dataset_list

def get_success_cnt(success_flag_list):
    success_flag_cnt = 0
    for success_flag_file in success_flag_list:
            if osp.exists(success_flag_file):
                success_flag_cnt += 1
    return success_flag_cnt

def main():
    success_flag_dir = "log/convert_success_flag"
    dataset_root = "/mnt/cfs/agi/data/pretrain/sun/auto_label_data/"
    dataset_list = get_process_order(dataset_root)
    success_flag_list = []

    for i,dataset in enumerate(dataset_list):
        success_flag = osp.join(success_flag_dir,f"{dataset}_SUCCESS")
        print(f"{i}, {dataset}")
        sleep_time = 0
        time.sleep(10)
        while (i-get_success_cnt(success_flag_list) > 8):
            sleep_time += 1
            time.sleep(10)
            print(f"{dataset} sleep {sleep_time/6.0} minites")
        if not osp.exists(success_flag):
            print(f"start process {dataset}")
            os.system(f'nohup python3.8 scripts/preprocess_autra/convert_autra_to_openscene_train_format.py --record_name={dataset} >log/dataset_convert_log/{dataset}.log 2>&1 &')
        else:
            print(f"{dataset} done!")
        success_flag_list.append(success_flag)
            

if __name__ == '__main__':
    main()

        
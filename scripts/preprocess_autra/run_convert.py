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


def main():
    dataset_root = "/mnt/cfs/agi/data/pretrain/sun/auto_label_data/"
    for i,dataset in enumerate(os.listdir(dataset_root)):
        #if dataset.startswith('autra'):
        if dataset in ["autra_train_m1.8", "autra_train_v2.25"]:
            #os.system(f'nohup python3.8 scripts/preprocess_autra/convert_autra_to_openscene_train_format.py --record_name={dataset} >dataset_convert_log/{dataset}.log 2>&1 &')
            print(f"dataset_convert_log/{dataset}.log")
            os.system(f'python3.8 scripts/preprocess_autra/convert_autra_to_openscene_train_format.py --record_name={dataset} >log/dataset_convert_log/{dataset}.log 2>&1 ')

if __name__ == '__main__':
    main()
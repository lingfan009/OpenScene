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



def main():
    dataset_list = ["autra_train_case_v1.0","autra_train_case_v1.1","autra_train_case_v1.10","autra_train_case_v1.11","autra_train_v2.25","autra_train_v2.27","autra_train_v2.32"]

    for i,dataset in enumerate(dataset_list):        
        os.system(f'nohup python3.8 scripts/preprocess_autra/test.py --record_name={dataset} >log/dataset_convert_log/{dataset}_test.log 2>&1 &')

            

if __name__ == '__main__':
    main()

        
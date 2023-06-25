import os
import time
import random
import numpy as np
import logging
import argparse
import os.path as osp
import sys
sys.path.append('/home/fan.ling/openscence/openscene_autra/OpenScene/scripts/preprocess')
from convert_autra_to_openscene_train_format import convert_autra_to_openscene_train_format

cur_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
log_file = f"./out/log/{cur_time}.txt"

def write_log(log_path, log_content, is_new_process = False):
    with open(log_path, 'a+') as f:
        if not is_new_process:
            cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            log_content = "time:{}, log_content:{}".format(cur_time, log_content)
        else:
            cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            log_content = "\n{} {}\n".format(cur_time, log_content)
        f.write(log_content)

def main():
    tos_root_path = "tos:camera-data-labeling-longmao/label_data_to_longmao"
    dataset_list = ["20230421_180000-longmao-uniform_sample-discrete-lidar_camera_union",
                    "20230329_180000-longmao-uniform_sample-discrete-lidar_camera_union",
                    "2023052001_180000-longmao-uniform_sample-discrete-lidar_camera_union"]

    distill_orig_train_data = "ori_sample_data/online_train"
    autra_train_batch_data = "./data/batch/"
    autra_train_data = "./data/"

    # download extract data
    for i,train_dataset in enumerate(dataset_list):
        # if i > 1:
        #     continue
        # download train_dataset
        dataset_path = osp.join(tos_root_path, train_dataset)
        os.system(f"rclone copy {dataset_path} {distill_orig_train_data} -P --update --fast-list --timeout=0 --transfers=48 --progress --s3-disable-checksum")

        # convert autra format to nuscenes format
        for record_name in os.listdir(distill_orig_train_data):
            record_path = osp.join(distill_orig_train_data, record_name)
            convert_autra_to_openscene_train_format(record_path, autra_train_batch_data)
        os.system(f"rm -rf {distill_orig_train_data}/*")

        # circle train model
        autra_train_batch_data_3d = osp.join(autra_train_batch_data, "nuscenes_autra_3d_test", "train")
        autra_train_batch_data_2d = osp.join(autra_train_batch_data, "nuscenes_autra_2d_test", "train")

        autra_train_batch_data_3d_size = len(os.listdir(autra_train_batch_data_3d))
        for index,scene_name in enumerate(os.listdir(autra_train_batch_data_3d)):
            scene_name = scene_name.split('.')[0]
            autra_train_data_3d = osp.join(autra_train_data, 'nuscenes_autra_3d_test', 'train')
            autra_val_data_3d = osp.join(autra_train_data, 'nuscenes_autra_3d_test', 'val')
            autra_train_data_2d = osp.join(autra_train_data, 'nuscenes_autra_2d_test', 'train')
            autra_val_data_2d = osp.join(autra_train_data, 'nuscenes_autra_2d_test', 'val')
            if not osp.exists(autra_train_data_3d):
                os.makedirs(autra_train_data_3d, exist_ok=True)
                os.makedirs(autra_val_data_3d, exist_ok=True)
                os.makedirs(autra_train_data_2d, exist_ok=True)
                os.makedirs(autra_val_data_2d, exist_ok=True)
            source_data_3d = osp.join(autra_train_batch_data_3d, scene_name+".pth")
            source_data_2d = osp.join(autra_train_batch_data_2d, scene_name)
            os.system(f"cp -rf {source_data_3d} {autra_train_data_3d}")
            os.system(f"cp -rf {source_data_3d} {autra_val_data_3d}")
            os.system(f"cp -rf {source_data_2d} {autra_train_data_2d}")
            os.system(f"cp -rf {source_data_2d} {autra_val_data_2d}")

            if (index+1) % 500 == 0 or (index+1) == autra_train_batch_data_3d_size:
                for ele in os.listdir(autra_train_data_3d):
                    log_content = f"batch_name:{autra_train_batch_data_3d}, index:{index+1}, scene_name:{ele.split('.')[0]}\n"
                    write_log(log_file, log_content)

                os.system(f"python scripts/feature_fusion/autra_openseg_multi_thread.py   --data_dir ./data  --output_dir ./fuse_2d_features/nuscenes_autra_2d_test  --openseg_model model_checkpoint/openseg_exported_clip  --split train")
                os.system(f"sh run/resume_distill.sh exp_dir/3d_distill_with_openseg_feature config/nuscenes/ours_openseg_pretrained_autra_data.yaml")   
                os.system(f"rm -rf {autra_train_data_3d}")
                os.system(f"rm -rf {autra_val_data_3d}")
                os.system(f"rm -rf {autra_train_data_2d}")
                os.system(f"rm -rf {autra_val_data_2d}")
                os.system(f"rm -rf ./fuse_2d_features/nuscenes_autra_2d_test")
                
        # rm batch_data
        os.system(f"rm -rf {autra_train_batch_data_3d}")
        os.system(f"rm -rf {autra_train_batch_data_2d}")


if __name__ == '__main__':
    main()

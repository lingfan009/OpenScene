from nuscenes.nuscenes import NuScenes
import torch
import os
import os.path as osp
import numpy as np
import pandas as pd
import struct
import open3d
import math
import imageio
import cv2
from scipy.spatial.transform import Rotation as R
from pypcd import pypcd
from plyfile import PlyData, PlyElement
import time

NUSCENES_FULL_CLASSES = ( # 32 classes
    'noise',
    'animal',
    'human.pedestrian.adult',
    'human.pedestrian.child',
    'human.pedestrian.construction_worker',
    'human.pedestrian.personal_mobility',
    'human.pedestrian.police_officer',
    'human.pedestrian.stroller',
    'human.pedestrian.wheelchair',
    'movable_object.barrier',
    'movable_object.debris',
    'movable_object.pushable_pullable',
    'movable_object.trafficcone',
    'static_object.bicycle_rack',
    'vehicle.bicycle',
    'vehicle.bus.bendy',
    'vehicle.bus.rigid',
    'vehicle.car',
    'vehicle.construction',
    'vehicle.emergency.ambulance',
    'vehicle.emergency.police',
    'vehicle.motorcycle',
    'vehicle.trailer',
    'vehicle.truck',
    'flat.driveable_surface',
    'flat.other',
    'flat.sidewalk',
    'flat.terrain',
    'static.manmade',
    'static.other',
    'static.vegetation',
    'vehicle.ego',
    'unlabeled',
)

VALID_NUSCENES_CLASS_IDS = ()

NUSCENES_CLASS_REMAP = 256*np.ones(32) # map from 32 classes to 16 classes
NUSCENES_CLASS_REMAP[2] = 7 # person
NUSCENES_CLASS_REMAP[3] = 7
NUSCENES_CLASS_REMAP[4] = 7
NUSCENES_CLASS_REMAP[6] = 7
NUSCENES_CLASS_REMAP[9] = 1 # barrier
NUSCENES_CLASS_REMAP[12] = 8 # traffic cone
NUSCENES_CLASS_REMAP[14] = 2 # bicycle
NUSCENES_CLASS_REMAP[15] = 3 # bus
NUSCENES_CLASS_REMAP[16] = 3
NUSCENES_CLASS_REMAP[17] = 4 # car
NUSCENES_CLASS_REMAP[18] = 5 # construction vehicle
NUSCENES_CLASS_REMAP[21] = 6 # motorcycle
NUSCENES_CLASS_REMAP[22] = 9 # trailer ???
NUSCENES_CLASS_REMAP[23] = 10 # truck
NUSCENES_CLASS_REMAP[24] = 11 # drivable surface
NUSCENES_CLASS_REMAP[25] = 12 # other flat??
NUSCENES_CLASS_REMAP[26] = 13 # sidewalk
NUSCENES_CLASS_REMAP[27] = 14 # terrain
NUSCENES_CLASS_REMAP[28] = 15 # manmade
NUSCENES_CLASS_REMAP[30] = 16 # vegetation

cur_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
log_file = f"/home/fan.ling/Work/big_model/openScene/openscene/log/{cur_time}.txt"

def extract_lidar_data(pcd_file_path):
    pcd = np.fromfile(pcd_file_path, dtype=np.float32)
    arr = pcd.reshape(-1, 5)
    seg_label = np.ones((arr.shape[0],1)) * -1
    #arr.tofile(pcd_file_path.replace(".bin", "_1.bin"))
    #print(arr.shape)
    return arr, seg_label

def save_lidar_data(lidar_data, export_all_points=True, save_dir=""):
    coords = np.ascontiguousarray(lidar_data[:, :3])
    category_id = np.ascontiguousarray(lidar_data[:, -1]).astype(int)

    category_id[category_id == -1] = 0
    remapped_labels = NUSCENES_CLASS_REMAP[category_id]
    remapped_labels -= 1
    torch.save((coords, 0, remapped_labels), save_dir)


def get_point_cloud_data(pc_path):
    with open(pc_path, 'rb') as f:
        plydata = PlyData.read(f)
        data = plydata.elements[0].data  # 读取数据
        data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
        property_names = data[0].dtype.names  # 读取property的名字

        data_np = np.zeros(data_pd.shape, dtype=np.float32)  # 初始化储存数据的array
        for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
            data_np[:, i] = data_pd[name]
        data_np[:, :3] = data_np[:, :3]

    return data_np

def write_log(log_path, log_content, is_new_process = False):
    with open(log_path, 'a+') as f:
        if not is_new_process:
            cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            log_content = "time:{}, log_content:{}".format(cur_time, log_content)
        else:
            cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            log_content = "\n{} {}\n".format(cur_time, log_content)
        f.write(log_content)

def convert_autra_to_openscene_format(train_data_tos_path, seg_train_data_tos_path, annos_name):
    # download train data
    train_data_local_path = "./perception_train_data/tos_data/"
    seg_train_data_local_path = "./perception_train_data/seg_data/"
    if not osp.exists(train_data_local_path):
        os.makedirs(train_data_local_path, exist_ok=True)
    if len(os.listdir(train_data_local_path)) == 0:
        os.system(f"rclone copy {train_data_tos_path}bins {train_data_local_path}bins --update --fast-list --timeout=0 --transfers=48 --progress --s3-disable-checksum")
        os.system(f"rclone copy {train_data_tos_path}new_bins {train_data_local_path}new_bins --update --fast-list --timeout=0 --transfers=48 --progress --s3-disable-checksum")
        os.system(f"rclone copy {train_data_tos_path}{annos_name} {train_data_local_path} --update --fast-list --timeout=0 --transfers=48 --progress --s3-disable-checksum")

    # convert to openscene format
    openscene_val_data_path = "./perception_train_data/openscene_val_data/"
    tos_bin_data_path = osp.join(train_data_local_path, "bins")
    if not os.path.exists(openscene_val_data_path):
        os.makedirs(openscene_val_data_path)
    if len(os.listdir(openscene_val_data_path)) == 0:
        for pcd_file in os.listdir(tos_bin_data_path):
            pcd_name = pcd_file.split(".")[0]
            pcd_file_path = osp.join(tos_bin_data_path, pcd_file)
            sample_point_data, sample_label_data = extract_lidar_data(pcd_file_path)
            save_data = np.concatenate([sample_point_data, sample_label_data], axis=1)
            save_dir = osp.join(openscene_val_data_path, pcd_name + ".pth")
            save_lidar_data(save_data, True, save_dir)

    # move to nuscenes_3d
    nuscenes_3d_path = "data/nuscenes_3d/val/"
    if not osp.exists(nuscenes_3d_path):
        os.makedirs(nuscenes_3d_path, exist_ok=True)
    if len(os.listdir(nuscenes_3d_path)) == 0:
        os.system(f"mv {openscene_val_data_path}* {nuscenes_3d_path}")

    # do evaluation to get seg label
    openscene_val_result_path = "out/nuscenes_openseg/autra_perception_train/val/distill"
    seg_label_result_path = f"{openscene_val_result_path}/result_eval/"
    if not osp.exists(seg_label_result_path):
        os.system(f"sh run/eval.sh {openscene_val_result_path} config/nuscenes/ours_openseg_pretrained_manual_autra.yaml distill")

    # merge seg label with original train data
    if not osp.exists(seg_label_result_path):
        os.makedirs(seg_label_result_path, exist_ok=True)
    if len(os.listdir(seg_label_result_path)) > 0:
        for seg_label_file in os.listdir(seg_label_result_path):
            if "labels" in seg_label_file:
                continue
            train_data_file = seg_label_file.split("_distill")[0] + ".bin"
            pcd_data = np.fromfile(osp.join(tos_bin_data_path, train_data_file), dtype=np.float32).reshape(-1, 5)
            seg_label = get_point_cloud_data(osp.join(seg_label_result_path,seg_label_file))

            pcd_coor = pcd_data[:,:3]
            pcd_intensity = pcd_data[:,3:4]
            pcd_ts = pcd_data[:,4:5]
            seg_label_coor = seg_label[:,:3]
            seg_label_color = seg_label[:,3:]
            pcd_save_data = np.concatenate([pcd_coor, pcd_intensity, seg_label_color, pcd_ts], axis=1)
            seg_save_local_path = osp.join(seg_train_data_local_path, "bins")
            if not osp.exists(seg_save_local_path):
                os.makedirs(seg_save_local_path, exist_ok=True)
            seg_pcd_save_path = osp.join(seg_save_local_path, train_data_file)
            pcd_save_data.astype(np.float32).tofile(seg_pcd_save_path)
        train_new_bins_data_local_path = osp.join(train_data_local_path, "new_bins")
        train_annos_data_local_path = osp.join(train_data_local_path, annos_name)
        os.system(f"cp -rf {train_new_bins_data_local_path} {seg_train_data_local_path}")
        os.system(f"cp -rf {train_annos_data_local_path} {seg_train_data_local_path}")

    '''
    # rclone to tos
    os.system(f"rclone copy {seg_train_data_local_path} {seg_train_data_tos_path} --update --fast-list --timeout=0 --transfers=48 --progress --s3-disable-checksum")

    # check and log
    train_data_local_path_bin = osp.join(train_data_local_path, "bins")
    seg_train_data_local_path = osp.join(seg_train_data_local_path, "bins")
    totol_cnt = 0
    mis_cnt = 0
    for item_file in os.listdir(train_data_local_path_bin):
        totol_cnt += 1
        seg_item_file = osp.join(seg_train_data_local_path, item_file)
        if not osp.exists(seg_item_file):
            mis_cnt += 1
            log_content = "version:{}, miss frame:{}!\n".format(train_data_tos_path, item_file)
            write_log(log_file, log_content)
    log_content = "finish-done-{}, version:{}, total_cnt:{}, mis_cnt:{}!\n".format(train_data_tos_path, train_data_tos_path, totol_cnt, mis_cnt)
    write_log(log_file, log_content)

    # delete local data
    os.system(f"rm -rf {train_data_local_path}")
    os.system(f"rm -rf {openscene_val_data_path}")
    os.system(f"rm -rf {nuscenes_3d_path}")
    os.system(f"rm -rf {openscene_val_result_path}")
    os.system(f"rm -rf {seg_train_data_local_path}")
    '''

train_data_version = train_data_root_list = [
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

def main():
    perception_train_data_root = "tos:perception-dataset-v2"
    perception_train_data_save_root = "tos:perception-dataset-v2/pre_labeled_dataset/self_learning"
    perception_train_data_save_root = "tos:data-labeling-huoshan/self_learning"
    #train_data_save_version = [version.replace("/", "_") for version in train_data_version]

    for version in train_data_version:
        if "autra_train_case_v1.0" in version:
            print(version)
            print(version.split("annos"))
            data_version = version.split("annos")[0]
            anno_name = version.split("/")[-1]
            train_data_path = osp.join(perception_train_data_root, data_version)
            save_data_path = osp.join(perception_train_data_save_root, data_version)
            # log
            log_content = "generating version:{}, !\n".format(data_version)
            write_log(log_file, log_content)
            # doing
            convert_autra_to_openscene_format(train_data_path, save_data_path, anno_name)


if __name__ == '__main__':
    #test_train_data()
    main()


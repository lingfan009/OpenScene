import torch
import os
import os.path as osp
import numpy as np
import struct
import open3d
import math
import imageio
import cv2
from scipy.spatial.transform import Rotation as R
from pypcd import pypcd
import json

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

# 将autra抽帧打包对齐的数据，转化成openscene支持的nuscene格式的训练数据集

cam_types = ['camera_upmiddle_right', 'camera_upmiddle_middle', 'camera_upmiddle_left', 'camera_left_front', 'camera_left_backward', 'camera_right_front', 'camera_right_backward']
cam_type_dict = {
    'camera_upmiddle_right': 'upmiddle_right_60h',
    'camera_upmiddle_middle': 'upmiddle_middle_120h',
    'camera_upmiddle_left': 'upmiddle_left_30h',
    'camera_left_front': 'left_front_120h',
    'camera_left_backward': 'left_backward_120h',
    'camera_right_front': 'right_front_120h',
    'camera_right_backward': 'right_backward_120h'
}
img_size = (800, 450)

def save_lidar_data(lidar_data, export_all_points=True, save_dir=""):
    coords = np.ascontiguousarray(lidar_data[:, :3])
    category_id = np.ascontiguousarray(lidar_data[:, -1]).astype(int)

    category_id[category_id == -1] = 0
    remapped_labels = NUSCENES_CLASS_REMAP[category_id]
    remapped_labels -= 1
    print(f"coords:{coords.shape}")
    torch.save((coords, 0, remapped_labels), save_dir)

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(
        image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic

def convert_autra_to_openscene_train_format(autra_train_data, openscene_train_data):
    for scene_index in os.listdir(autra_train_data):
        scene_name = os.listdir(osp.join(autra_train_data, scene_index))[0]
        lidar_dir = osp.join(autra_train_data, scene_index, scene_name, "lidar")
        lidar_file = osp.join(lidar_dir, os.listdir(lidar_dir)[0])

        # save lidar
        try:
            pcd = pypcd.PointCloud.from_path(lidar_file).pc_data
        except:
            print("error for ", lidar_file)

        coors = np.concatenate([
            pcd['y'], -pcd['x'], pcd['z'], pcd['intensity'],
            pcd['timestamp']]).astype(np.float32).reshape(5, -1).T
        valid_point_mask = (coors[:, 0] > 0) | (coors[:, 1] > 0)
        coors = coors[valid_point_mask]

        labels = np.ones((coors.shape[0], 1))*17
        save_data = np.concatenate([coors, labels], axis=1)
        save_dir = osp.join(openscene_train_data, 'nuscenes_autra_3d_test', 'train')
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        save_file = osp.join(openscene_train_data, 'nuscenes_autra_3d_test', 'train', scene_name + ".pth")
        save_lidar_data(save_data, True, save_file)

        # create folder
        out_dir_color = os.path.join(openscene_train_data, 'nuscenes_autra_2d_test', 'train', scene_name, 'color')
        out_dir_pose = os.path.join(openscene_train_data, 'nuscenes_autra_2d_test', 'train', scene_name, 'pose')
        out_dir_K = os.path.join(openscene_train_data, 'nuscenes_autra_2d_test', 'train', scene_name, 'K')
        os.makedirs(out_dir_color, exist_ok=True)
        os.makedirs(out_dir_pose, exist_ok=True)
        os.makedirs(out_dir_K, exist_ok=True)

        # extract meta
        meta_file = osp.join(autra_train_data, scene_index, scene_name, 'msg_meta.json')
        meta_info = json.load(open(meta_file, "r"))
        sync_meta = meta_info['sync_meta']
        for camera_type in cam_types:
            cur_sync_meta = None
            for sync_meta_ele in sync_meta:
                if cam_type_dict[camera_type] in sync_meta_ele['frame_id']:
                    cur_sync_meta = sync_meta_ele
            k = cur_sync_meta['intrinsics']
            pose = cur_sync_meta['extrinsics']

            camera_dir = osp.join(autra_train_data, scene_index, scene_name, camera_type)
            camera_file = osp.join(camera_dir, os.listdir(camera_dir)[0])

            # save image
            img = imageio.v3.imread(camera_file)
            img_shape = img.shape
            img = cv2.resize(img, img_size)
            imageio.imwrite(os.path.join(out_dir_color, camera_type + '.jpg'), img)

            # copy the camera parameters to the folder
            rotation_i2c = pose['value'][3:7]
            rot_matrix_i2c = R.from_quat(rotation_i2c).as_matrix()
            trans_matrix_i2c = np.array(pose['value'][0:3]).reshape(-1, 1)


            padding_matrix = np.array([[0, 0, 0, 1]])
            pose = np.concatenate([np.concatenate([rot_matrix_i2c, trans_matrix_i2c], axis=1), padding_matrix], axis=0)
            pose = np.linalg.inv(pose)
            np.save(os.path.join(out_dir_pose, camera_type + '.npy'), pose)

            # k
            camera_intrinsic = np.asarray(k)
            K = adjust_intrinsic(camera_intrinsic, intrinsic_image_dim=(img_shape[1], img_shape[0]), image_dim=img_size)
            np.save(os.path.join(out_dir_K, camera_type + '.npy'), K)


def main():
    autra_train_data = "ori_sample_data/Robin-20230511_131509_1683784739_1683784769"
    openscene_train_data = "data/"
    convert_autra_to_openscene_train_format(autra_train_data, openscene_train_data)

if __name__ == '__main__':
    main()

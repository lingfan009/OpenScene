import os
import os.path as osp
import pdb

import torch
import argparse
import numpy as np
from pypcd import pypcd
from glob import glob
from tqdm import tqdm, trange

def get_args():
    '''Command line arguments.'''
    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of Mask2Former on Autra.')
    #parser.add_argument('--mask2former_input_dir', type=str, help=' ')
    #parser.add_argument('--output_dir', type=str, help='')
    #parser.add_argument('--split', type=str, default='test', help='split: "train"| "val" ')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args

def get_point_cloud_from_pcd(pc_path):
    lidar_points = pypcd.PointCloud.from_path(pc_path)
    data_np = np.concatenate(
        [lidar_points.pc_data['x'], lidar_points.pc_data['y'], 
        lidar_points.pc_data['z'], lidar_points.pc_data['intensity'], lidar_points.pc_data['label']]).reshape(5, -1).T
    return data_np


def process_one_scene(data_path, out_dir, text_embeddings):
    '''Process one scene.'''

    # load 3D data (point cloud, color and the corresponding labels)
    # Only process points with GT label annotation
    #load point_with_label and filter padding points(0,0,0)
    points_with_label = get_point_cloud_from_pcd(data_path)
    valid_point_mask = ~((points_with_label[:, 0] == 0) & (points_with_label[:, 1] == 0) & (points_with_label[:, 2] == 0))
    points_with_label = points_with_label[valid_point_mask]
    points_bg_selected = points_with_label >= 500
    points_with_label[points_bg_selected] = 0 
    mask_entire = points_with_label[:,4] <= 5

    # # fill text embedding features
    points_with_feature = np.zeros((points_with_label.shape[0], 768))
    labels = points_with_label[:, 4].astype(np.int16)
    points_with_feature[:] = text_embeddings[labels]

    #pdb.set_trace()
    # save point feature
    points_with_feature_torch = torch.from_numpy(points_with_feature)
    scene_id = osp.basename(data_path).replace("-lidar.pcd", ".pth")
    torch.save({"feat": points_with_feature_torch.half().cpu(), "mask_full": mask_entire}, osp.join(out_dir, scene_id))

def get_clip_text_embedding(class_name_list):
    text_embedding_path = "/home/fan.ling/big_model/OpenScene/OpenScene/fuse_2d_features/nuscenes_autra_2d_test/text_embedding_feature_6_cls.pth"
    text_embedding_feature = torch.load(text_embedding_path)["text_embedding_feature"].numpy()
    return text_embedding_feature

def main(args):
    class_names = ['background', 'car', 'person', 'bicycle', "traffic cone", "road"]
    extract_dir_list = ["Robin_20230517_164224"]
    autra_detect_dem_label_dir = "/mnt/cfs/agi/workspace/LidarAnnotation/data/"
    openscene_autra_train_dir = "/home/fan.ling/big_model/OpenScene/OpenScene/fuse_2d_features/nuscenes_autra_2d_test/autra_detect_dem_label/"

    text_embedding = get_clip_text_embedding(class_names)
    for extract_dir in extract_dir_list:
        autra_detect_dem_label_record = osp.join(autra_detect_dem_label_dir, extract_dir, "lidar")
        for frame_name in tqdm(os.listdir(autra_detect_dem_label_record)):
            frame_path = osp.join(autra_detect_dem_label_record, frame_name)
            process_one_scene(frame_path, openscene_autra_train_dir, text_embedding)

if __name__ == "__main__":
    args = get_args()
    print(f"Arguments:{args}")
    main(args)

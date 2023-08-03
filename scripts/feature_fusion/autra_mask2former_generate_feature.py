import os
import os.path as osp
import pdb

import torch
import argparse
import numpy as np
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


def process_one_scene(data_path, out_dir, text_embeddings):
    '''Process one scene.'''

    # load 3D data (point cloud, color and the corresponding labels)
    # Only process points with GT label annotation

    #load point_with_label and filter padding points(0,0,0)
    coors = np.fromfile(data_path, dtype=np.float32).reshape(-1,5) 
    valid_point_mask = ~((coors[:, 0] == 0) & (coors[:, 1] == 0) & (coors[:, 2] == 0))
    points_with_label = coors[valid_point_mask]
    mask_entire = points_with_label[:,4] != 0

    # fill text embedding features
    points_with_feature = np.zeros((points_with_label.shape[0], 768))
    labels = points_with_label[:, 4].astype(np.int16)
    points_with_feature[:] = text_embeddings[labels]

    # save point feature
    points_with_label_feature = np.concatenate([points_with_label, points_with_feature], axis=1) 
    points_with_label_feature_torch = torch.from_numpy(points_with_label_feature)
    points_with_feature_torch = torch.from_numpy(points_with_feature)
    scene_id = osp.basename(data_path).replace("-lidar.bin", ".pth")
    print(points_with_feature_torch.shape)
    torch.save({"feat": points_with_feature_torch.half().cpu(), "mask_full": mask_entire}, osp.join(out_dir, scene_id))



def get_clip_text_embedding(class_name_list):
    text_embedding_path = "/home/fan.ling/big_model/OpenScene/OpenScene/fuse_2d_features/nuscenes_autra_2d_test/text_embedding_feature.pth"
    text_embedding_feature = torch.load(text_embedding_path)["text_embedding_feature"].numpy()

    return text_embedding_feature


def main(args):
    class_names = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water', 'window-blind', 'window', 'tree', 'fence', 'ceiling', 'sky', 'cabinet', 'table', 'floor', 'pavement', 'mountain', 'grass', 'dirt', 'paper', 'food', 'building', 'rock', 'wall', 'rug']
    extract_dir_list = ["Robin_20230517_164224"]
    mask2former_label_dir = "/mnt/cfs/agi/workspace/LidarAnnotation/data/"
    openscene_autra_train_dir = "/home/fan.ling/big_model/OpenScene/OpenScene/fuse_2d_features/nuscenes_autra_2d_test/mask2former_batch1/"

    text_embedding = get_clip_text_embedding(class_names)
    i = 0
    for extract_dir in extract_dir_list:
        mask2former_label_record = osp.join(mask2former_label_dir, extract_dir, "lidar")
        for frame_name in tqdm(os.listdir(mask2former_label_record)):
            frame_path = osp.join(mask2former_label_record, frame_name)
            if i<0 or "1684314319521" in frame_path or True:
                process_one_scene(frame_path, openscene_autra_train_dir, text_embedding)
            i += 1

if __name__ == "__main__":
    args = get_args()
    print(f"Arguments:{args}")

    main(args)

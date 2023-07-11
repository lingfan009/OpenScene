import os
import pdb

import torch
import argparse
from os.path import join, exists
import numpy as np
import time
import imageio
from glob import glob
from tqdm import tqdm, trange
import tensorflow as tf2
import tensorflow.compat.v1 as tf
from concurrent.futures import ThreadPoolExecutor
import threading
from fusion_util import extract_ovseg_img_feature, PointCloudToImageMapperV1

config_file = "/home/fan.ling/big_model/OvSeg/OvSeg/configs/ovseg_swinB_vitL_demo.yaml"
class_names = 'car,tree,grass,pole,road,cyclist,vehicle,truck,bicycle,other flat,buildings,safety barriers,sidewalk,manmade,sky,bus,suv,person,rider'
class_names = class_names.split(',')
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 
               'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
               'tennis racket' 'couch', 'potted plant', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'banner', 'blanket', 'bridge', 
               'net', 'pillow', 'platform', 'playingfield', 'railroad', 'road',  'stairs', 'tent', 'towel', 'water', 
               'tree', 'fence', 'ceiling', 'sky', 'cabinet', 'mountain', 'grass', 'dirt',  'building', 'rock', 'wall', 'rug']
model_weights = 'MODEL.WEIGHTS /home/fan.ling/big_model/OvSeg/OvSeg/checkpoints/ovseg_swinbase_vitL14_ft_mpt.pth'

def get_args():
    '''Command line arguments.'''
    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on nuScenes.')
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--split', type=str, default='test', help='split: "train"| "val" ')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args


def process_one_scene(data_path, out_dir, args):
    '''Process one scene.'''

    # short hand
    split = args.split
    data_root_2d = args.data_root_2d
    point2img_mapper = args.point2img_mapper
    cam_locs = ['camera_upmiddle_right', 'camera_upmiddle_middle', 'camera_upmiddle_left', 'camera_left_front', 'camera_left_backward', 'camera_right_front', 'camera_right_backward']

    # load 3D data (point cloud, color and the corresponding labels)
    # Only process points with GT label annotation
    locs_in = torch.load(data_path)[0]
    labels_in = torch.load(data_path)[2]
    mask_entire = labels_in!=255

    locs_in = locs_in[mask_entire]
    n_points = locs_in.shape[0]

    scene_id = data_path.split('/')[-1].split('.')[0]
    if exists(join(out_dir, scene_id +'.pt')):
        print(scene_id +'.pt' + ' already done!')
        return 1

    # process 2D features
    scene = join(data_root_2d, split, scene_id)
    img_dir_base = join(scene, 'color')
    pose_dir_base = join(scene, 'pose')
    K_dir_base = join(scene, 'K')
    num_img = len(cam_locs)

    device = torch.device('cpu')
    n_points_cur = n_points
    counter = torch.zeros((n_points_cur, 1), device=device)
    sum_features = torch.zeros((n_points_cur, args.feat_dim), device=device)


    vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=device)
    text_features_final = None
    cam_weight_dict = {
        'camera_upmiddle_right': 5, 
        'camera_upmiddle_middle': 1, 
        'camera_upmiddle_left': 3, 
        'camera_left_front': 1, 
        'camera_left_backward': 3, 
        'camera_right_front': 1, 
        'camera_right_backward': 3
    }
    for img_id, cam in enumerate(tqdm(cam_locs)):
        # if not (cam == "camera_upmiddle_right" or cam == "camera_upmiddle_left" or cam == "camera_upmiddle_middle"):
        #     continue
        # extract 2d feat    
        intr = np.load(join(K_dir_base, cam+'.npy'))
        pose = np.load(join(pose_dir_base, cam+'.npy'))
        img_dir = join(img_dir_base, cam+'.jpg')

        img = imageio.v3.imread(img_dir)
        img_shape = (img.shape[1], img.shape[0])
        feat_2ds, text_features = extract_ovseg_img_feature(config_file, scene_id, cam, class_names, img_dir, model_weights)

        # calculate the 3d-2d mapping
        mapping = np.ones([n_points_cur, 4], dtype=int)
        mapping[:, 1:4] = point2img_mapper.compute_mapping(img_shape, scene_id, cam, pose, locs_in, depth=None, intrinsic=intr)
        if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
            continue

        mapping = torch.from_numpy(mapping).to(device)
        mask = mapping[:, 3]
        vis_id[:, img_id] = mask
        feat_2d = feat_2ds.permute(2, 0, 1)
        feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0)

        counter[mask!=0]+= cam_weight_dict[cam]
        sum_features[mask!=0] += feat_2d_3d.cpu()[mask!=0] * cam_weight_dict[cam]
        text_features_final = text_features

    counter[counter==0] = 1e-5
    feat_bank = sum_features/counter
    point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])

    mask = torch.zeros(n_points, dtype=torch.bool)
    mask[point_ids] = True
    mask_entire[mask_entire==True] = mask
    point_cloud_label = torch.argmax((feat_bank.cpu() @ text_features_final.cpu().T)[:, :-1], dim=1).reshape(-1,1)
    locs_in = torch.from_numpy(locs_in)
    point_cloud_label = torch.cat((locs_in, point_cloud_label), 1)

    torch.save({"point_with_label": point_cloud_label[mask].cpu()}, join(out_dir, "point_with_label", scene_id +'_with_label.pt'))
    torch.save({"feat": feat_bank[mask].half().cpu(), "mask_full": mask_entire}, join(out_dir, "point_with_feature", scene_id +'.pt'))
    print(join(out_dir, scene_id +'.pt') + ' is saved!')


def process_one_scene1(params):
    process_one_scene(params[0], params[1], params[2])


def main(args):
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    args.cut_num_pixel_boundary = 5 # do not use the features on the image boundary
    args.feat_dim = 768 # CLIP feature dimension
    split = args.split
    data_dir = args.data_dir

    data_root = join(data_dir, 'nuscenes_autra_3d_test')
    data_root_2d = join(data_dir,'nuscenes_autra_2d_test')

    args.data_root_2d = data_root_2d
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # calculate image pixel-3D points correspondances
    args.point2img_mapper = PointCloudToImageMapperV1(
            cut_bound=args.cut_num_pixel_boundary)

    data_paths = sorted(glob(join(data_root, split, '*.pth')))
    total_num = len(data_paths)
    
    for i in trange(total_num):
        process_one_scene(data_paths[i], out_dir, args)

    # # multi thread handle
    # with ThreadPoolExecutor(max_workers=max_workers) as pool:
    #     #data_paths_list = []
    #     for i in trange(total_num):
    #         # data_paths_list.append((data_paths[i], out_dir, args, i))
    #         # if len(data_paths_list) == 1 or (i+1) == total_num:
    #         #     beg = time.perf_counter()
    #         #     results = pool.map(process_one_scene1, data_paths_list)
    #         #     data_paths_list.clear()
    #         #     end = time.perf_counter()
    #         #     print("multi time use: ",end-beg)
    #         process_one_scene(data_paths[i], out_dir, args)

if __name__ == "__main__":
    args = get_args()
    print(f"Arguments:{args}")

    main(args)

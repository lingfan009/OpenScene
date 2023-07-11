import os
import argparse
import torch

from plyfile import PlyData, PlyElement
import numpy as np
import pandas as pd
from pypcd import pypcd
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

def get_point_cloud_data(pc_path):
    if ".bin" in pc_path:
        pcd = np.fromfile(pc_path, dtype=np.float32)
        arr = pcd.reshape(-1, 8)
        data_np = arr[:, [0,1,2,4,5,6]]
    else:
        with open(pc_path, 'rb') as f:
            plydata = PlyData.read(f)
            data = plydata.elements[0].data  # 读取数据
            data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
            property_names = data[0].dtype.names  # 读取property的名字

            data_np = np.zeros(data_pd.shape, dtype=np.float)  # 初始化储存数据的array
            for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
                data_np[:, i] = data_pd[name]
            data_np[:, :3] = data_np[:, :3]

    return data_np

def parse_args():
    args = argparse.ArgumentParser(description="Data processor for data labeling pipeline.")
    args.add_argument('--feature_type', type=str, default='input', help='input file')
    args.add_argument('--pcd_index', type=str, help='1~8')
    return args.parse_args()

def visualize_segment_pcd_file(distill_pcd_file):
    print(distill_pcd_file)
    # 读取对应点云
    data_np = get_point_cloud_data(distill_pcd_file)
    # visualization
    app = gui.Application.instance
    app.initialize()

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(data_np[:, :3])
    cloud.colors = o3d.utility.Vector3dVector(data_np[:, 3:] / 255)
    # vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 2048, 1536)
    vis.show_settings = True
    vis.add_geometry("Points", cloud)
    app.add_window(vis)
    app.run()

def draw_point(feature_type, pcd_index):
    # 获取要可视化文件
    # distill_pcd_dir = f"/home/fan.ling/Work/big_model/openScene/openscene/out/nuscenes_openseg/autra_perception_train/val/{feature_type}/result_eval/"
    # pcd_index = int(pcd_index)
    # if pcd_index >= len(os.listdir(distill_pcd_dir)):
    #     pcd_index = len(os.listdir(distill_pcd_dir))-1
    # file_name = os.listdir(distill_pcd_dir)[pcd_index]
    # #distill_pcd_file = f"/home/fan.ling/Work/big_model/openScene/openscene/out/nuscenes_openseg/autra_perception_train/val/{feature_type}/result_eval/{pcd_index}_{feature_type}.ply"
    # distill_pcd_file = f"/home/fan.ling/Work/big_model/openScene/openscene/out/nuscenes_openseg/autra_perception_train/val/{feature_type}/result_eval/{file_name}"
    distill_pcd_file = f"../test_data/45908478058883072-1655205302326.bin"

    distill_pcd_file = f"./test_data/1669709785020-Robin.bin"
    distill_pcd_file = f"./test_data/1669710197020-Robin.bin"
    distill_pcd_file = f"./test_data/case_1/bins/1671416131610-Robin.bin"
    distill_pcd_root = f"./test_data/case_1/bins"
    #print(distill_pcd_file)
    index = 0
    for distill_pcd_file in os.listdir(distill_pcd_root):
        index += 1
        if distill_pcd_file not in ("1672294295610-ACRush.bin", "1672294301120-ACRush.bin", "1672294298610-ACRush.bin"):
            continue
        print(index)
        distill_pcd_file = os.path.join(distill_pcd_root, distill_pcd_file)
        visualize_segment_pcd_file(distill_pcd_file)


def visual_point_cloud_with_label():
    data_path = "/home/fan.ling/big_model/OpenScene/OpenScene/fuse_2d_features/nuscenes_autra_2d_test/point_with_label/1687768892023-Robin_with_label.pt"
    data_np = torch.load(data_path)['point_with_label']

    class_names = 'car,tree,grass,pole,road,cyclist,vehicle,truck,bicycle,other flat,buildings,safety barriers,sidewalk,manmade,sky,bus,suv,person,rider'
    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 
               'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
               'tennis racket' 'couch', 'potted plant', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'banner', 'blanket', 'bridge', 
               'net', 'pillow', 'platform', 'playingfield', 'railroad', 'road',  'stairs', 'tent', 'towel', 'water', 
               'tree', 'fence', 'ceiling', 'sky', 'cabinet', 'mountain', 'grass', 'dirt',  'building', 'rock', 'wall', 'rug']
    
    show_class_list = ['car', 'truck', 'bus']
    data_np_list = []
    for class_name in show_class_list:
        class_index = class_names.index(class_name)
        data_np_list.append(data_np[data_np[:,3].int() == class_index])
    data_np = torch.concat(data_np_list, dim=0)

    # visualization
    app = gui.Application.instance
    app.initialize()

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(data_np[:, :3])
    #cloud.colors = o3d.utility.Vector3dVector(np.ones((data_np.shape[0], 3))*data_np[:, 3:4]*12 / 255)
    cloud.colors = o3d.utility.Vector3dVector(np.ones((data_np.shape[0], 3)) * 5 * 12 / 255)
    # vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 2048, 1536)

    vis.show_settings = True
    vis.add_geometry("Points", cloud)
    app.add_window(vis)
    app.run()

if __name__ == "__main__":
    args = parse_args()
    #draw_point(args.feature_type, args.pcd_index)
    visual_point_cloud_with_label()
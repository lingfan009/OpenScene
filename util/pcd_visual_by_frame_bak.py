import os
import argparse

from plyfile import PlyData, PlyElement
import numpy as np
import pandas as pd
from pypcd import pypcd
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

def get_point_cloud_data(pc_path):
    with open(pc_path, 'rb') as f:
        plydata = PlyData.read(f)
        data = plydata.elements[0].data  # 读取数据
        data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
        property_names = data[0].dtype.names  # 读取property的名字

        # print(property_names)
        # print(data_pd['red'])
        data_np = np.zeros(data_pd.shape, dtype=np.float)  # 初始化储存数据的array
        for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
            data_np[:, i] = data_pd[name]
        # print(data_np)
        #data_np_color = data_np[:, 3:]
        #uniques = np.unique(data_np_color, axis=0)
        #print(uniques)
        # data_np_intensity = data_np_color[:, 0] + data_np_color[:, 1] + data_np_color[:, 2]
        # max_intensity = np.max(data_np_intensity)
        # scale_factor = 255 / max_intensity
        # data_np_intensity = data_np_intensity * scale_factor

        data_np[:, :3] = data_np[:, :3]
        # x_center = (np.max(data_np[:, 0]) - np.min(data_np[:, 0]))/2
        # y_center = (np.max(data_np[:, 1]) - np.min(data_np[:, 1])) / 2
        # z_center = (np.max(data_np[:, 2]) - np.min(data_np[:, 2])) / 2
        # data_np[:, 0] = data_np[:, 0] - x_center
        # data_np[:, 1] = data_np[:, 1] - y_center
        # data_np[:, 2] = data_np[:, 2] - z_center
        # #data_np[:, 2] = data_np_intensity
        #
        # pcd_save_data = data_np[:, :5]
        # #print(pcd_save_data)
        # bin_path = "/home/fan.ling/Work/big_model/openScene/openscene/out/replica_openseg/result_eval_2d/7_input.bin"
        # pcd_save_data.astype(np.float32).tofile(bin_path)

        # pcd_path = "/home/fan.ling/Work/big_model/openScene/openscene/out/replica_openseg/result_eval_2d/7_input.pcd"
        # # pcd_data.astype(np.float32).tofile(lidar_index_path)
        # pypcd.save_point_cloud(pcd_save_data, pcd_path)
    return data_np

def parse_args():
    args = argparse.ArgumentParser(description="Data processor for data labeling pipeline.")
    args.add_argument('--feature_type', type=str, default='input', help='input file')
    args.add_argument('--pcd_index', type=str, help='1~8')
    return args.parse_args()

def draw_point(feature_type, pcd_index):
    property_pcd_file = f"/home/fan.ling/Work/big_model/openScene/openscene/out/replica_openseg/result_eval_property/{pcd_index}_ensemble.ply"
    distill_pcd_file = f"/home/fan.ling/Work/big_model/openScene/openscene/out/replica_openseg/result_eval_3d/{pcd_index}_distill.ply"
    fusion_pcd_file = f"/home/fan.ling/Work/big_model/openScene/openscene/out/replica_openseg/result_eval_2d/{pcd_index}_fusion.ply"
    ensemble_pcd_file = f"/home/fan.ling/Work/big_model/openScene/openscene/out/replica_openseg/result_eval_ensemble/{pcd_index}_ensemble.ply"
    input_pcd_file = f"/home/fan.ling/Work/big_model/openScene/openscene/out/replica_openseg/result_eval_2d/{pcd_index}_input.ply"

    distill_pcd_file = f"/home/fan.ling/Work/big_model/openScene/openscene/out/nuscenes/distill/result_eval/{pcd_index}_distill.ply"

    distill_pcd_file = f"/home/fan.ling/Work/big_model/openScene/openscene/out/nuscenes/autra/val/distill/result_eval/{pcd_index}_distill.ply"

    data_np_distill = get_point_cloud_data(distill_pcd_file)
    data_np_fusion = get_point_cloud_data(fusion_pcd_file)
    data_np_input = get_point_cloud_data(input_pcd_file)
    data_np_ensemble = get_point_cloud_data(ensemble_pcd_file)
    data_np_property = get_point_cloud_data(property_pcd_file)

    if feature_type == "fusion":
        data_np = data_np_fusion
    elif feature_type == "distill":
        data_np = data_np_distill
    elif feature_type == "ensemble":
        data_np = data_np_ensemble
    elif feature_type == "property":
        data_np = data_np_property
    else:
        data_np = data_np_input

    # visualization
    app = gui.Application.instance
    app.initialize()

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(data_np[:, :3])
    cloud.colors = o3d.utility.Vector3dVector(data_np[:, 3:]/255)
    #vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True
    vis.add_geometry("Points", cloud)
    app.add_window(vis)
    app.run()

if __name__ == "__main__":
    args = parse_args()
    draw_point(args.feature_type, args.pcd_index)
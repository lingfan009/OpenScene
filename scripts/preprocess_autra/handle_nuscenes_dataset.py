from nuscenes.nuscenes import NuScenes
import torch
import os.path as osp
import numpy as np
import struct
import open3d

# import torch
# from os.path import join
# import numpy as np
#
# datapath = "/home/fan.ling/Work/big_model/openScene/openscene/data/nuscenes_3d/val"
# processed_data = torch.load(join(datapath, "64b2cd57232f4358992168b0952c9e15.pth"))
# print(processed_data)
# print(len(processed_data[2]))
# print(len(processed_data[0]))
# data_np_color = processed_data[2]
# uniques = np.unique(data_np_color, axis=0)
# print(uniques)

# nuscenes lidar数据存储格式 ：(x,y,z,intensity,ring index)
def read_bin_velodyne(path):
    pc_list=[]
    with open(path,'rb') as f:
        content=f.read()
        pc_iter=struct.iter_unpack('fffff',content)
        for idx,point in enumerate(pc_iter):
            #print(idx, point)
            pc_list.append([point[0],point[1],point[2]])
    return np.asarray(pc_list,dtype=np.float32)

def read_label_bin_velodyne(path):
    pc_list=[]
    with open(path,'rb') as f:
        content=f.read()
        print(content)
        pc_iter=struct.iter_unpack('f',content)
        for idx,point in enumerate(pc_iter):
            print(idx, point)
            pc_list.append([point[0],point[1],point[2]])
    return np.asarray(pc_list,dtype=np.float32)


def main():
    # nusc = NuScenes(version='v1.0-trainval',
    #                 dataroot='/home/fan.ling/Work/big_model/openScene/openscene/original_dataset/nuscenes/',
    #                 verbose=True)

    nusc = NuScenes(version='v1.0-mini',
                     dataroot='/home/fan.ling/Work/big_model/openScene/openscene/original_dataset/nuscenes/v1.0-mini_orig/',
                     verbose=True)
    scenes_list = nusc.list_scenes()
    scenes_list = nusc.scene

    #nusc.list_lidarseg_categories(sort_by='count')

    for i in range(len(scenes_list)):
        if i != 0:
            continue
        scene = scenes_list[i]
        first_sample_token = scene['first_sample_token']
        #print(scene)
        my_sample = nusc.get('sample', first_sample_token)
        nusc.get_sample_lidarseg_stats(my_sample['token'], sort_by='count')
        print(nusc.lidarseg_idx2name_mapping)
        next_sample_token = my_sample['next']
        # sample_data_token = my_sample['data']['LIDAR_TOP']
        # nusc.render_sample_data(sample_data_token,
        #                         with_anns=False,
        #                         show_lidarseg=True)
        #
        # nusc.render_sample_data(sample_data_token,
        #                         with_anns=True,
        #                         show_lidarseg=True,
        #                         filter_lidarseg_labels=[17, 23])
        #
        # nusc.render_sample_data(sample_data_token,
        #                         with_anns=False,
        #                         show_lidarseg=True,
        #                         show_lidarseg_legend=True)
        # blog : https://bbs.huaweicloud.com/blogs/349393

        # nusc.render_pointcloud_in_image(my_sample['token'],
        #                                 pointsensor_channel='LIDAR_TOP',
        #                                 camera_channel='CAM_FRONT',
        #                                 render_intensity=False,
        #                                 show_lidarseg=True,
        #                                 filter_lidarseg_labels=[17, 23, 24],
        #                                 show_lidarseg_legend=True)

        j = 0
        while next_sample_token != "":
            print(j)
            next_sample = nusc.get('sample', next_sample_token)
            next_sample_token = next_sample['next']
            #print(my_sample)
            j += 1
            for sensor_type in next_sample['data']:
                cam_data = nusc.get('sample_data', next_sample['data'][sensor_type])
                #print(cam_data)
                print("cam_type:{}, channel_type:{}, file_name:{}".format(cam_data['sensor_modality'], cam_data['channel'], cam_data['filename']) )
                #nusc.render_sample_data(cam_data['token'])
                calibrated_sensor = nusc.get("calibrated_sensor",  cam_data['calibrated_sensor_token'])
                #print(calibrated_sensor)
                ego_pose = nusc.get("ego_pose",  cam_data['ego_pose_token'])
                #print(ego_pose)
                #lidarseg = nusc.get("lidarseg", cam_data['filename'])
                #print(lidarseg)
                if cam_data['sensor_modality'] == "lidar":
                    lidar_pcd_path = osp.join("./original_dataset/nuscenes/v1.0-mini_orig/", cam_data['filename'])
                    print(cam_data)
                    # processed_data = torch.load(osp.join("./original_dataset/nuscenes/v1.0-mini_orig/", cam_data['filename']))
                    pcd = open3d.open3d.geometry.PointCloud()

                    processed_data = read_bin_velodyne(lidar_pcd_path)
                    print(processed_data.shape)
                    pcd.points = open3d.open3d.utility.Vector3dVector(processed_data)
                    open3d.open3d.visualization.draw_geometries([pcd])

                    lidar_label_path = osp.join("./original_dataset/nuscenes/v1.0-mini_orig/lidarseg/v1.0-mini", cam_data['token'] + "_lidarseg.bin")
                    #label_data_torch = torch.load(lidar_label_path)
                    b = np.fromfile(lidar_label_path, dtype=np.uint8)
                    print(b.shape)

                    #label_data = read_label_bin_velodyne(lidar_label_path)
                    NUSCENES_CLASS_REMAP = 256 * np.ones(32)
                    print(NUSCENES_CLASS_REMAP)


if __name__ == '__main__':
    main()
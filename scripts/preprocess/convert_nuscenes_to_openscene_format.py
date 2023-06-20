from nuscenes.nuscenes import NuScenes
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


split = 'train'  # 'train' | 'val'
out_3d_dir = 'data/nuscenes_3d/{}'.format(split)
out_2d_dir = 'data/nuscenes_2d/{}'.format(split)

os.makedirs(out_3d_dir, exist_ok=True)
os.makedirs(out_2d_dir, exist_ok=True)

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


cam_locs = ['back', 'back_left', 'back_right', 'front', 'front_left', 'front_right']

cam_locs = {
    "CAM_BACK": 'back',
    "CAM_BACK_LEFT": 'back_left',
    "CAM_BACK_RIGHT": 'back_right',
    "CAM_FRONT": 'front',
    "CAM_FRONT_LEFT": 'front_left',
    "CAM_FRONT_RIGHT": 'front_right'
}
img_size = (800, 450)

# nuscenes lidar数据存储格式 ：(x,y,z,intensity,ring index)
def read_bin_velodyne(path):
    pc_list=[]
    with open(path,'rb') as f:
        content=f.read()
        pc_iter=struct.iter_unpack('fffff',content)
        for idx,point in enumerate(pc_iter):
            #print(idx, point)
            pc_list.append([point[0],point[1],point[2],point[3]])
            #pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list,dtype=np.float32)

# lidar seg label 数据存储格式：(category_id)
def read_label_bin_velodyne(path):
    pc_list=[]
    with open(path,'rb') as f:
        content=f.read()
        pc_iter=struct.iter_unpack('f',content)
        for idx,point in enumerate(pc_iter):
            pc_list.append([point[0],point[1],point[2]])
    return np.asarray(pc_list,dtype=np.float32)

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

def extract_camera_data(camera_data, nusc, scene_name):

    out_dir_color = os.path.join(out_2d_dir, scene_name, 'color')
    out_dir_pose = os.path.join(out_2d_dir, scene_name, 'pose')
    out_dir_K = os.path.join(out_2d_dir, scene_name, 'K')
    os.makedirs(out_dir_color, exist_ok=True)
    os.makedirs(out_dir_pose, exist_ok=True)
    os.makedirs(out_dir_K, exist_ok=True)

    print(camera_data)
    cam_type = camera_data['channel']
    cam_type = cam_locs[cam_type]
    cam_img_path = osp.join("./original_dataset/nuscenes/v1.0-mini_orig/", camera_data['filename'])

    img = imageio.v3.imread(cam_img_path)
    img = cv2.resize(img, img_size)
    imageio.imwrite(os.path.join(out_dir_color, cam_type + '.jpg'), img)


    # copy the camera parameters to the folder
    calibrated_sensor = nusc.get("calibrated_sensor",  camera_data['calibrated_sensor_token'])
    rotation = calibrated_sensor['rotation']
    rot_matrix = R.from_quat(rotation).as_matrix()
    trans_matrix = np.array(calibrated_sensor['translation']).reshape(-1,1)
    padding_matrix = np.array([[0, 0, 0, 1]])
    pose = np.concatenate([np.concatenate([rot_matrix,trans_matrix], axis=1),padding_matrix], axis=0)
    np.save(os.path.join(out_dir_pose, cam_type + '.npy'), pose)

    # k
    print(calibrated_sensor)
    camera_intrinsic = calibrated_sensor['camera_intrinsic']
    camera_intrinsic = np.asarray(camera_intrinsic)
    K = adjust_intrinsic(camera_intrinsic, intrinsic_image_dim=(1600, 900), image_dim=img_size)
    np.save(os.path.join(out_dir_K, cam_type + '.npy'), K)
    #     K_dir = os.path.join(data_path, scene, 'frames', timestamp, cam, 'K.txt')
    #     K = np.asarray([[float(x[0]), float(x[1]), float(x[2])] for x in
    #                     (x.split(" ") for x in open(K_dir).read().splitlines())])
    #     K = adjust_intrinsic(K, intrinsic_image_dim=(1600, 900), image_dim=img_size)
    #     np.save(os.path.join(out_dir_K, cam + '.npy'), K)
    #
    #     # shutil.copyfile(pose_dir, os.path.join(out_dir_K, cam+'.txt'))
    # pass

def extract_lidar_data(lidar_data):
    # read lidar point cloud data
    lidar_pcd_path = osp.join("./original_dataset/nuscenes/v1.0-mini_orig/", lidar_data['filename'])
    #pcd = open3d.open3d.geometry.PointCloud()
    lidar_pcd_data = read_bin_velodyne(lidar_pcd_path)
    #pcd.points = open3d.open3d.utility.Vector3dVector(lidar_pcd_data)
    #open3d.open3d.visualization.draw_geometries([pcd])

    # read lidar seg label
    lidar_label_path = osp.join("./original_dataset/nuscenes/v1.0-mini_orig/lidarseg/v1.0-mini",
                                lidar_data['token'] + "_lidarseg.bin")
    #print(lidar_label_path)
    lidar_label_data = np.fromfile(lidar_label_path, dtype=np.uint8)
    lidar_label_data = lidar_label_data.reshape((-1,1))
    return lidar_pcd_data, lidar_label_data

def save_lidar_data(lidar_data, export_all_points=True, save_dir=""):
    coords = np.ascontiguousarray(lidar_data[:, :3])
    category_id = np.ascontiguousarray(lidar_data[:, -1]).astype(int)

    category_id[category_id == -1] = 0
    remapped_labels = NUSCENES_CLASS_REMAP[category_id]
    remapped_labels -= 1
    print(f"coords:{coords.shape}")
    torch.save((coords, 0, remapped_labels), save_dir)

def main():
    nusc = NuScenes(version='v1.0-mini',
                     dataroot='/home/fan.ling/Work/big_model/openScene/openscene/original_dataset/nuscenes/v1.0-mini_orig/',
                     verbose=True)
    scenes_list = nusc.scene

    for i in range(len(scenes_list)):
        # if i != 0:
        #     continue
        scene = scenes_list[i]
        #scene_name = scene['token']
        #print(scene)
        #print(f"scene_name:{scene_name}")
        next_sample_token = scene['first_sample_token']

        j = 0
        # scene_points_data = []
        # scene_labels_data = []
        while next_sample_token != "":
            #print(j)
            my_sample = nusc.get('sample', next_sample_token)
            next_sample_token = my_sample['next']
            scene_name =  my_sample['token']
            j += 1
            for sensor_type in my_sample['data']:
                cam_data = nusc.get('sample_data', my_sample['data'][sensor_type])
                if cam_data['sensor_modality'] == "lidar":
                    sample_point_data, sample_label_data = extract_lidar_data(cam_data)
                    # scene_points_data.append(sample_point_data)
                    # scene_labels_data.append(sample_label_data)

                if cam_data['sensor_modality'] == "camera":
                    extract_camera_data(cam_data, nusc, scene_name)
            # merge_points_data = np.concatenate(scene_points_data, axis=0)
            # merge_labels_data = np.concatenate(scene_labels_data, axis=0)
            save_data = np.concatenate([sample_point_data, sample_label_data], axis=1)
            save_dir = osp.join(out_3d_dir, scene_name+".pth")
            save_lidar_data(save_data, True, save_dir)


def test_train_data():
    data_path = "/home/fan.ling/Work/big_model/openScene/openscene/data/nuscenes_3d/val/95be92f4ae2f41cbb460e84b2665ad9f.pth"
    pcd_data = torch.load(data_path)
    print(pcd_data[0].shape)


if __name__ == '__main__':
    #test_train_data()
    main()

import math
import numpy as np
import os
import os.path as osp

def main():
    autra_train_data_3d = "/home/fan.ling/big_model/OpenScene/OpenScene/data/nuscenes_autra_3d_test/train/"
    autra_train_data_3d_filter = "/home/fan.ling/big_model/OpenScene/OpenScene/data/nuscenes_autra_3d_test_mask2former/train/"
    autra_train_data_2d = "/home/fan.ling/big_model/OpenScene/OpenScene/fuse_2d_features/nuscenes_autra_2d_test/mask2former_batch1/"

    for file_name in os.listdir(autra_train_data_2d):
        #file_dir = osp.join(autra_train_data_3d, file_name.replace("pt","pth"))
        file_dir = osp.join(autra_train_data_3d, file_name)
        os.system(f"cp -rf {file_dir} {autra_train_data_3d_filter}")
        

if __name__ == '__main__':
    main()
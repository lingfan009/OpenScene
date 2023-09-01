'''Dataloader for fused point features.'''

import copy
import os
import pdb
from glob import glob
from os.path import join
import torch
import numpy as np
import SharedArray as SA
import tensorflow as tf2
import tensorflow.compat.v1 as tf

from dataset.point_loader import Point3DLoader, AutraPoint3DLoader
import sys
sys.path.append('/home/fan.ling/big_model/OpenScene/OpenScene/scripts/feature_fusion')
from fusion_util import PointCloudToImageMapper
from autra_ovseg import process_one_scene_online

class FusedFeatureLoader(AutraPoint3DLoader):
    '''Dataloader for fused point features.'''

    def __init__(self,
                 datapath_prefix,
                 datapath_prefix_feat,
                 train_dataset_list=[], eval_dataset_list = [], 
                 voxel_size=0.05,
                 split='train', aug=False, memcache_init=False,
                 identifier=7791, loop=1, eval_all=False,
                 input_feature = False,
                 eval_type = "distill"
                 ):
        super().__init__(datapath_prefix=datapath_prefix, train_dataset_list=train_dataset_list, 
                         eval_dataset_list=eval_dataset_list,voxel_size=voxel_size,
                                           split=split, aug=aug, memcache_init=memcache_init,
                                           identifier=identifier, loop=loop,
                                           eval_all=eval_all, input_feature=input_feature)
        self.aug = aug
        self.input_feature = input_feature # decide whether we use point color values as input
        self.eval_type = eval_type
        # prepare for 3D features
        self.datapath_feat = datapath_prefix_feat
        # Precompute the occurances for each scene
        # for training sets, ScanNet and Matterport has 5 each, nuscene 1
        # for evaluation/test sets, all has just one
        if 'nuscenes' in self.dataset_name or "autra" in self.dataset_name: # only one file for each scene
            self.list_occur = None
        if len(self.data_paths) == 0:
            raise Exception('0 file is loaded in the feature loader.')

    def __getitem__(self, index_long):

        index = index_long % len(self.data_paths)
        if self.use_shm:
            locs_in = SA.attach("shm://%s_%s_%06d_locs_%08d" % (
                self.dataset_name, self.split, self.identifier, index)).copy()
            feats_in = SA.attach("shm://%s_%s_%06d_feats_%08d" % (
                self.dataset_name, self.split, self.identifier, index)).copy()
            labels_in = SA.attach("shm://%s_%s_%06d_labels_%08d" % (
                self.dataset_name, self.split, self.identifier, index)).copy()
        else:
            locs_in, feats_in, labels_in = torch.load(self.data_paths[index])
            #print(f" labels_in:{np.unique(labels_in, return_counts=True)}")
            labels_in[labels_in == 999] = 255
            labels_in = labels_in.astype(np.uint8)
            if np.isscalar(feats_in) and feats_in == 0:
                # no color in the input point cloud, e.g nuscenes lidar
                feats_in = np.zeros_like(locs_in)
            else:
                feats_in = feats_in

        # load 3D features
        scene_name = self.data_paths[index][:-4].split('/')[-1]
        dataset_version = self.data_paths[index].split('/')[-3]
        processed_data = torch.load(join(self.datapath_feat, dataset_version, scene_name+'.pth'))

        flag_mask_merge = False
        if len(processed_data.keys())==2:
            flag_mask_merge = True
            feat_3d, mask_chunk = processed_data['feat'], processed_data['mask_full']
            if isinstance(mask_chunk, np.ndarray): # if the mask itself is a numpy array
                mask_chunk = torch.from_numpy(mask_chunk)
            mask = copy.deepcopy(mask_chunk)
            if self.split != 'train': # val or test set
                mask_chunk = torch.ones_like(mask_chunk) # every point needs to be evaluted
        elif len(processed_data.keys())>2: # legacy, for old processed features
            feat_3d, mask_visible, mask_chunk = processed_data['feat'], processed_data['mask'], processed_data['mask_full']
            mask = torch.zeros(feat_3d.shape[0], dtype=torch.bool)
            mask[mask_visible] = True # mask out points without feature assigned

        if len(feat_3d.shape)>2:
            feat_3d = feat_3d[..., 0]

        locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in

        # calculate the corresponding point features after voxelization
        if self.split == 'train' and flag_mask_merge:
            #print("enter_1")
            locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                locs_in, feats_in, labels_in, return_ind=True)
            vox_ind = torch.from_numpy(vox_ind)
            mask = mask_chunk[vox_ind] # voxelized visible mask for entire point cloud
            mask_ind = mask_chunk.nonzero(as_tuple=False)[:, 0]
            index1 = - torch.ones(mask_chunk.shape[0], dtype=int)
            index1[mask_ind] = mask_ind

            index1 = index1[vox_ind]
            chunk_ind = index1[index1!=-1]

            index2 = torch.zeros(mask_chunk.shape[0])
            index2[mask_ind] = 1
            index3 = torch.cumsum(index2, dim=0, dtype=int)
            # get the indices of corresponding masked point features after voxelization
            indices = index3[chunk_ind] - 1

            # get the corresponding features after voxelization
            feat_3d = feat_3d[indices]
        else:
            locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                locs[mask_chunk], feats_in[mask_chunk], labels_in[mask_chunk], return_ind=True)
            vox_ind = torch.from_numpy(vox_ind)
            feat_3d = feat_3d[vox_ind]
            mask = mask[vox_ind]

        if self.eval_all: # during evaluation, no voxelization for GT labels
            labels = labels_in
        if self.aug:
            locs, feats, labels = self.input_transforms(locs, feats, labels)
        coords = torch.from_numpy(locs).int()
        coords = torch.cat((torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
        if self.input_feature:
            feats_zeros = torch.zeros(coords.shape[0], 3)
            feats_intensity = torch.from_numpy(feats).float() / 256.0
            feats_zeros[:,0] = feats_intensity[:,0]
            feats = feats_zeros
        else:
            # hack: directly use color=(1, 1, 1) for all points
            feats = torch.ones(coords.shape[0], 3)
        labels = torch.from_numpy(labels).long()

        if self.eval_all:
            return coords, feats, labels, feat_3d, mask, torch.from_numpy(inds_reconstruct).long()
        return coords, feats, labels, feat_3d, mask

def collation_fn(batch):
    '''
    :param batch:
    :return:    coords: N x 4 (batch,x,y,z)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)

    '''
    coords, feats, labels, feat_3d, mask_chunk = list(zip(*batch))

    for i in range(len(coords)):
        coords[i][:, 0] *= i

    return torch.cat(coords), torch.cat(feats), torch.cat(labels), \
        torch.cat(feat_3d), torch.cat(mask_chunk)


def collation_fn_eval_all(batch):
    '''
    :param batch:
    :return:    coords: N x 4 (x,y,z,batch)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)
                inds_recons:ON

    '''
    coords, feats, labels, feat_3d, mask, inds_recons = list(zip(*batch))
    inds_recons = list(inds_recons)

    accmulate_points_num = 0
    for i in range(len(coords)):
        coords[i][:, 0] *= i
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]

    return torch.cat(coords), torch.cat(feats), torch.cat(labels), \
        torch.cat(feat_3d), torch.cat(mask), torch.cat(inds_recons)

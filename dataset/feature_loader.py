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

from dataset.point_loader import Point3DLoader
import sys
sys.path.append('/home/fan.ling/openscence/openscene_autra/OpenScene/scripts/feature_fusion')
from fusion_util import PointCloudToImageMapper
from autra_ovseg import process_one_scene_online

class FusedFeatureLoader(Point3DLoader):
    '''Dataloader for fused point features.'''

    def __init__(self,
                 datapath_prefix,
                 datapath_prefix_feat,
                 voxel_size=0.05,
                 split='train', aug=False, memcache_init=False,
                 identifier=7791, loop=1, eval_all=False,
                 input_color = False,
                 eval_type = "distill"
                 ):
        super().__init__(datapath_prefix=datapath_prefix, voxel_size=voxel_size,
                                           split=split, aug=aug, memcache_init=memcache_init,
                                           identifier=identifier, loop=loop,
                                           eval_all=eval_all, input_color=input_color)
        self.aug = aug
        self.input_color = input_color # decide whether we use point color values as input
        self.eval_type = eval_type
        # prepare for 3D features
        self.datapath_feat = datapath_prefix_feat

        # Precompute the occurances for each scene
        # for training sets, ScanNet and Matterport has 5 each, nuscene 1
        # for evaluation/test sets, all has just one
        if 'nuscenes' in self.dataset_name: # only one file for each scene
            self.list_occur = None
        else:
            self.list_occur = []
            #print(f"self.data_paths:{self.data_paths}")
            for data_path in self.data_paths:
                if 'scannet' in self.dataset_name:
                    scene_name = data_path[:-15].split('/')[-1]
                else:
                    scene_name = data_path[:-4].split('/')[-1]
                    scene_name = data_path[:-4].split('/')[-1]
                file_dirs = glob(join(self.datapath_feat, scene_name + '_*.pt'))
                self.list_occur.append(len(file_dirs))
            #print(f"self.list_occur:{self.list_occur}")
            # some scenes in matterport have no features at all
            ind = np.where(np.array(self.list_occur) != 0)[0]
            if np.any(np.array(self.list_occur)==0):
                data_paths, list_occur = [], []
                for i in ind:
                    data_paths.append(self.data_paths[i])
                    list_occur.append(self.list_occur[i])
                self.data_paths = data_paths
                self.list_occur = list_occur

        # online load feature
        seed = 1457
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        self.img_dim = (800, 450)
        self.saved_model_path = '/home/fan.ling/openscence/openscene_autra/OpenScene/model_checkpoint/openseg_exported_clip'

        self.openseg_model = tf2.saved_model.load(self.saved_model_path,
                        tags=[tf.saved_model.tag_constants.SERVING],)
        self.text_emb = tf.zeros([1, 1, 768])
        self.feat_dim = 768

        # calculate image pixel-3D points correspondances
        self.point2img_mapper = PointCloudToImageMapper(
                image_dim=self.img_dim,
                cut_bound=5)

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
            labels_in[labels_in == -100] = 255
            labels_in = labels_in.astype(np.uint8)
            if np.isscalar(feats_in) and feats_in == 0:
                # no color in the input point cloud, e.g nuscenes lidar
                feats_in = np.zeros_like(locs_in)
            else:
                feats_in = (feats_in + 1.) * 127.5

        # load 3D features
        if self.dataset_name == 'scannet_3d':
            scene_name = self.data_paths[index][:-15].split('/')[-1]
        else:
            scene_name = self.data_paths[index][:-4].split('/')[-1]

        if 'nuscenes' not in self.dataset_name:
            n_occur = self.list_occur[index]
            if n_occur > 1:
                nn_occur = np.random.randint(n_occur)
            elif n_occur == 1:
                nn_occur = 0
            else:
                raise NotImplementedError

            processed_data = torch.load(join(
                self.datapath_feat, scene_name+'_%d.pt'%(nn_occur)))
        # elif "nuscenes_autra" in self.dataset_name:
        #     # online inference feature
        #     #print(self.data_paths[index])
        #     #processed_data = torch.load(join(self.datapath_feat, scene_name+'.pt'))
        #     # data/nuscenes_autra_3d_test/train/1683784749023-Robin.pth

        #     data_path = self.data_paths[index]
        #     param_dict = {}
        #     param_dict['split'] = self.split
        #     param_dict['img_dim'] = self.img_dim
        #     param_dict['data_root_2d'] = os.path.join(data_path.split("nuscenes_autra_3d_test")[0], "nuscenes_autra_2d_test")
        #     param_dict['point2img_mapper'] = self.point2img_mapper
        #     param_dict['openseg_model'] = self.openseg_model
        #     param_dict['text_emb'] = self.text_emb
        #     param_dict['feat_dim'] = self.feat_dim

        #     processed_data = process_one_scene_online(data_path, param_dict)
            
        else:
            # no repeated file
            processed_data = torch.load(join(self.datapath_feat, scene_name+'.pt'))
            # if self.eval_type == "distill":
            #     #fusion_featue_path = join(self.datapath_feat, os.listdir(self.datapath_feat)[0])
            #     fusion_featue_path = join(self.datapath_feat,scene_name+'.pt')
            #     processed_data = torch.load(fusion_featue_path)
            #     mask_default = torch.zeros(locs_in.shape[0], dtype=torch.bool).numpy()
            #     mask_full = processed_data['mask_full']
            #     mask_shape = mask_full.shape[0]
            #     mask_default[:] = False
            #     mask_default[:mask_shape] =  mask_full
            #     processed_data["mask_full"] = mask_default
            # else:
            #     processed_data = torch.load(join(self.datapath_feat, scene_name+'.pt'))
            #     mask_default = torch.zeros(locs_in.shape[0], dtype=torch.bool).numpy()
            #     mask_full = processed_data['mask_full']
            #     mask_shape = mask_full.shape[0]
            #     mask_default[:] = False
            #     mask_default[:mask_shape] =  mask_full
            #     processed_data["mask_full"] = mask_default


        flag_mask_merge = False
        if len(processed_data.keys())==2:
            flag_mask_merge = True
            feat_3d, mask_chunk = processed_data['feat'], processed_data['mask_full']
            if isinstance(mask_chunk, np.ndarray): # if the mask itself is a numpy array
                mask_chunk = torch.from_numpy(mask_chunk)
            mask = copy.deepcopy(mask_chunk)
            if self.split != 'train': # val or test set
                feat_3d_new = torch.zeros((locs_in.shape[0], feat_3d.shape[1]), dtype=feat_3d.dtype)
                feat_3d_new[mask] = feat_3d
                feat_3d = feat_3d_new
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
        elif self.split == 'train' and not flag_mask_merge: # legacy, for old processed features
            #print("enter_2")
            feat_3d = feat_3d[mask] # get features for visible points
            locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                locs_in, feats_in, labels_in, return_ind=True)
            mask_chunk[mask_chunk.clone()] = mask
            vox_ind = torch.from_numpy(vox_ind)
            mask = mask_chunk[vox_ind] # voxelized visible mask for entire point clouds
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
            #print("enter_3")
            #print("mask_chunk:", mask_chunk)
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
        if self.input_color:
            feats = torch.from_numpy(feats).float() / 127.5 - 1.
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

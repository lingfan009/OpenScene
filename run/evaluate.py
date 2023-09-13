import os
import random
import numpy as np
import logging
import argparse
import urllib
import pdb

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from util import metric
from torch.utils import model_zoo
from pypcd import pypcd

from MinkowskiEngine import SparseTensor
from util import config
from util.util import export_pointcloud, get_palette, \
    convert_labels_with_palette, extract_text_feature, visualize_labels
from tqdm import tqdm
from run.distill import get_model

from dataset.label_constants import *


def get_parser():
    '''Parse the config file.'''

    parser = argparse.ArgumentParser(description='OpenScene evaluation')
    parser.add_argument('--config', type=str,
                    default='config/scannet/eval_openseg.yaml',
                    help='config file')
    parser.add_argument('opts',
                    default=None,
                    help='see config/scannet/test_ours_openseg.yaml for all options',
                    nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    '''Define logger.'''

    logger_name = "main-logger"
    logger_in = logging.getLogger(logger_name)
    logger_in.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(filename)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger_in.addHandler(handler)
    return logger_in

def is_url(url):
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')

def main_process():
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def precompute_text_related_properties(labelset_name):
    '''pre-compute text features, labelset, palette, and mapper.'''

    if 'scannet' in labelset_name:
        labelset = list(SCANNET_LABELS_20)
        labelset[-1] = 'other' # change 'other furniture' to 'other'
        palette = get_palette(colormap='scannet')
    elif labelset_name == 'matterport_3d' or labelset_name == 'matterport':
        labelset = list(MATTERPORT_LABELS_21)
        palette = get_palette(colormap='matterport')
    elif labelset_name == 'nuscenes_autra_3d_test_mask2former':
        labelset = list(PROPERTY_INFOS)
        palette = get_palette(colormap='property_info')
    elif 'matterport_3d_40' in labelset_name or labelset_name == 'matterport40':
        labelset = list(MATTERPORT_LABELS_40)
        palette = get_palette(colormap='matterport_160')
    elif 'matterport_3d_80' in labelset_name or labelset_name == 'matterport80':
        labelset = list(MATTERPORT_LABELS_80)
        palette = get_palette(colormap='matterport_160')
    elif 'matterport_3d_160' in labelset_name or labelset_name == 'matterport160':
        labelset = list(MATTERPORT_LABELS_160)
        palette = get_palette(colormap='matterport_160')
    elif 'nuscenes_autra_3d_dataset' in labelset_name:
        labelset = list(NUSCENES_LABELS_6)
        palette = get_palette(colormap='nuscenes6')
    elif 'nuscenes' in labelset_name:
        labelset = list(NUSCENES_LABELS_6)
        palette = get_palette(colormap='nuscenes6')
    else: # an arbitrary dataset, just use a large labelset
        labelset = list(MATTERPORT_LABELS_160)
        palette = get_palette(colormap='matterport_160')

    mapper = None
    if hasattr(args, 'map_nuscenes_details'):
        labelset = list(NUSCENES_LABELS_DETAILS)
        mapper = torch.tensor(MAPPING_NUSCENES_DETAILS, dtype=int)
    text_features = extract_text_feature(labelset, args)
    return text_features, labelset, mapper, palette

def main():
    '''Main function.'''

    args = get_parser()

    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed    
    args.ngpus_per_node = len(args.test_gpu)
    if len(args.test_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False

    
    # By default we do not use shared memory for evaluation
    if not hasattr(args, 'use_shm'):
        args.use_shm = False
    if args.use_shm:
        if args.multiprocessing_distributed:
            args.world_size = args.ngpus_per_node * args.world_size
            mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.test_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    model = get_model(args)
    if main_process():
        global logger
        logger = get_logger()
        logger.info(args)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
        args.test_workers = int(args.test_workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
    else:
        model = model.cuda()

    if args.feature_type == 'fusion':
        pass # do not need to load weight
    elif is_url(args.model_path): # load from url
        checkpoint = model_zoo.load_url(args.model_path, progress=True)
        model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()},  strict=True)
        #model.load_state_dict(checkpoint['state_dict'], strict=True)
    
    elif args.model_path is not None and os.path.isfile(args.model_path):
        # load from directory
        if main_process():
            logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage.cuda())
        try:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        except Exception as ex:
            # The model was trained in a parallel manner, so need to be loaded differently
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('module.'):
                    # remove module
                    k = k[7:]
                else:
                    # add module
                    k = 'module.' + k

                new_state_dict[k]=v
            model.load_state_dict(new_state_dict, strict=True)
            logger.info('Loaded a parallel model')

        if main_process():
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))    
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    # ####################### Data Loader ####################### #
    if not hasattr(args, 'input_feature'):
        # by default we do not use the point color as input
        args.input_feature = False
    
    from dataset.feature_loader import FusedFeatureLoader, collation_fn_eval_all
    val_data = FusedFeatureLoader(datapath_prefix=args.data_root,
                                datapath_prefix_feat=args.data_root_2d_fused_feature,
                                train_dataset_list=args.train_dataset_list,
                                eval_dataset_list=args.eval_dataset_list,
                                voxel_size=args.voxel_size, 
                                split=args.split, aug=False,
                                memcache_init=args.use_shm, eval_all=True, identifier=6797,
                                input_feature=args.input_feature)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size,
                                                shuffle=False, num_workers=args.test_workers, pin_memory=True,
                                                drop_last=False, collate_fn=collation_fn_eval_all,
                                                sampler=val_sampler)

    # ####################### Test ####################### #
    labelset_name = args.data_root.split('/')[-1]
    if hasattr(args, 'labelset'):
        # if the labelset is specified
        labelset_name = args.labelset
    evaluate(model, val_loader, labelset_name)


def make_xyz_rgb_label_point_cloud(xyz_rgb_lable, metadata=None):
    """ Make a pointcloud object from xyz array.
    xyz array is assumed to be float32.
    rgb is assumed to be encoded as float32 according to pcl conventions.
    """
    md = {'version': .7,
          'fields': ['x', 'y', 'z', 'rgb', 'label'],
          'count': [1, 1, 1, 1, 1],
          'width': len(xyz_rgb_lable),
          'height': 1,
          'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          'points': len(xyz_rgb_lable),
          'type': ['F', 'F', 'F', 'U', 'U'],
          'size': [4, 4, 4, 4, 4],
          'data': 'binary'}
    if xyz_rgb_lable.dtype != np.float32:
        raise ValueError('array must be float32')
    if metadata is not None:
        md.update(metadata)

    xyz_rgb_lable_copy = xyz_rgb_lable.copy()
    rgb = xyz_rgb_lable[:,3].astype(np.uint32)
    label = xyz_rgb_lable[:,4].astype(np.uint32)
    
    pc_data = xyz_rgb_lable_copy.view(np.dtype([('x', np.float32),
                                     ('y', np.float32),
                                     ('z', np.float32),
                                     ('rgb', np.uint32),
                                     ('label', np.uint32)])).squeeze()
    pc_data['rgb']= rgb
    pc_data['label']= label
    pc = pypcd.PointCloud(md, pc_data)
    return pc

def convert_to_sustech_format(pcl, logits_pred, pcd_save_file, palette):
    
    label_len = int(len(palette)/3)
    color_map = np.ones((label_len))
    for i in range(label_len):
        color_map[i] = (palette[i*3 + 0] << 16) + (palette[i*3 + 1] << 8) + (palette[i*3 + 2] << 0)
    pcl_color = color_map[logits_pred].reshape((-1,1))
    pcl_cls = logits_pred.reshape((-1,1))
    save_data_np = np.concatenate([pcl, pcl_color, pcl_cls], axis=1)
    
    save_data_np_fp32 = save_data_np.astype(dtype=np.float32)
    
    save_data_pcd = make_xyz_rgb_label_point_cloud(save_data_np_fp32)
    # save pcd
    save_data_pcd.save_pcd(pcd_save_file, compression='binary')
    # save bin
    #save_data_np.astype(np.float32).tofile(pcd_save_file)

def evaluate(model, val_data_loader, labelset_name='scannet_3d'):
    '''Evaluate our OpenScene model.'''

    torch.backends.cudnn.enabled = False

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder, exist_ok=True)

    if args.save_feature_as_numpy: # save point features to folder
        out_root = os.path.commonprefix([args.save_folder, args.model_path])
        saved_feature_folder = os.path.join(out_root, 'saved_feature')
        os.makedirs(saved_feature_folder, exist_ok=True)

    # short hands
    save_folder = args.save_folder
    feature_type = args.feature_type
    eval_iou = True
    if hasattr(args, 'eval_iou'):
        eval_iou = args.eval_iou
    mark_no_feature_to_unknown = False
    if hasattr(args, 'mark_no_feature_to_unknown') and args.mark_no_feature_to_unknown and feature_type == 'fusion':
        # some points do not have 2D features from 2D feature fusion. Directly assign 'unknown' label to those points during inference
        mark_no_feature_to_unknown = True
    vis_input = False
    if hasattr(args, 'vis_input') and args.vis_input:
        vis_input = True
    vis_pred = False
    if hasattr(args, 'vis_pred') and args.vis_pred:
        vis_pred = True
    vis_gt = False
    if hasattr(args, 'vis_gt') and args.vis_gt:
        vis_gt = True

    text_features, labelset, mapper, palette = \
        precompute_text_related_properties(labelset_name)

    with torch.no_grad():
        model.eval()
        store = 0.0
        for rep_i in range(args.test_repeats):
            preds, gts = [], []
            val_data_loader.dataset.offset = rep_i
            if main_process():
                logger.info(
                    "\nEvaluation {} out of {} runs...\n".format(rep_i+1, args.test_repeats))

            # repeat the evaluation process
            # to account for the randomness in MinkowskiNet voxelization
            if rep_i>0:
                seed = np.random.randint(10000)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            if mark_no_feature_to_unknown:
                masks = []

            for i, (coords, feat, label, feat_3d, mask, inds_reverse) in enumerate(tqdm(val_data_loader)):
                sinput = SparseTensor(feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
                coords = coords[inds_reverse, :]
                pcl = coords[:, 1:].cpu().numpy()
                if feature_type == 'distill':
                    predictions = model(sinput)
                    predictions = predictions[inds_reverse, :]
                    text_features = text_features.to(predictions.device)
                    pred = predictions.half() @ text_features.half().t()
                    logits_pred = torch.max(pred, 1)[1].cpu()
                    #print(logits_pred.shape, logits_pred)
                elif feature_type == 'fusion':
                    predictions = feat_3d.cuda(non_blocking=True)[inds_reverse, :]
                    pred = predictions.half() @ text_features.t()
                    logits_pred = torch.max(pred, 1)[1].detach().cpu()
                    if mark_no_feature_to_unknown:
                    # some points do not have 2D features from 2D feature fusion.
                    # Directly assign 'unknown' label to those points during inference.
                        logits_pred[~mask[inds_reverse]] = len(labelset)-1
                elif feature_type == 'ensemble':
                    feat_fuse = feat_3d.cuda(non_blocking=True)[inds_reverse, :]
                    # pred_fusion = feat_fuse.half() @ text_features.t()
                    pred_fusion = (feat_fuse/(feat_fuse.norm(dim=-1, keepdim=True)+1e-5)).half() @ text_features.t()

                    predictions = model(sinput)
                    predictions = predictions[inds_reverse, :]
                    pred_distill = (predictions/(predictions.norm(dim=-1, keepdim=True)+1e-5)).half() @ text_features.t()
                    feat_ensemble = predictions.clone().half()
                    mask_ =  pred_distill.max(dim=-1)[0] < pred_fusion.max(dim=-1)[0]
                    feat_ensemble[mask_] = feat_fuse[mask_]
                    pred = feat_ensemble @ text_features.t()
                    logits_pred = torch.max(pred, 1)[1].detach().cpu()

                    predictions = feat_ensemble # if we need to save the features
                else:
                    raise NotImplementedError

                if args.save_feature_as_numpy:
                    scene_name = val_data_loader.dataset.data_paths[i].split('/')[-1].split('.pth')[0]
                    np.save(os.path.join(saved_feature_folder, '{}_openscene_feat_{}.npy'.format(scene_name, feature_type)), predictions.cpu().numpy())
                
                # Visualize the input, predictions and GT

                # special case for nuScenes, evaluation points are only a subset of input
                if 'nuscenes' in labelset_name:
                    label_mask = (label!=255)
                    label = label[label_mask]
                    logits_pred = logits_pred[label_mask]
                    pred = pred[label_mask]
                    if vis_pred:
                        print(val_data_loader.dataset.data_paths[i])
                        pcl = torch.load(val_data_loader.dataset.data_paths[i])[0][label_mask]

                if vis_input:
                    input_color = torch.load(val_data_loader.dataset.data_paths[i])[1]
                    export_pointcloud(os.path.join(save_folder, '{}_input.ply'.format(i)), pcl, colors=(input_color+1)/2)

                if vis_pred:
                    scene_name = val_data_loader.dataset.data_paths[i].split('/')[-1].split('.pth')[0]
                    if mapper is not None:
                        pred_label_color = convert_labels_with_palette(mapper[logits_pred].numpy(), palette)
                        export_pointcloud(os.path.join(save_folder, '{}_{}.ply'.format(scene_name, feature_type)), pcl, colors=pred_label_color)
                    else:
                        pred_label_color = convert_labels_with_palette(logits_pred.numpy(), palette)
                        export_pointcloud(os.path.join(save_folder, '{}_{}.ply'.format(scene_name, feature_type)), pcl, colors=pred_label_color)
                        visualize_labels(list(np.unique(logits_pred.numpy())),
                                    labelset,
                                    palette,
                                    os.path.join(save_folder, '{}_labels_{}.jpg'.format(i, feature_type)), ncol=5)
                
                # convert_to_sustech_format
                scene_name = val_data_loader.dataset.data_paths[i].split('/')[-1].split('.pth')[0]
                pcd_save_path = os.path.join(f"/mnt/cfs/agi/workspace/LidarAnnotation/data/{labelset_name}_prod_sign_eval/lidar")
                if not os.path.exists(pcd_save_path):
                    os.makedirs(pcd_save_path, exist_ok=True)
                pcd_save_file = os.path.join(pcd_save_path, scene_name + ".pcd")
                convert_to_sustech_format(pcl, logits_pred.numpy(), pcd_save_file, palette)

                # Visualize GT labels
                if vis_gt:
                    # for points not evaluating
                    label[label==255] = len(labelset)-1
                    gt_label_color = convert_labels_with_palette(label.cpu().numpy(), palette)
                    export_pointcloud(os.path.join(save_folder, '{}_gt.ply'.format(i)), pcl, colors=gt_label_color)
                    visualize_labels(list(np.unique(label.cpu().numpy())),
                                labelset,
                                palette,
                                os.path.join(save_folder, '{}_labels_gt.jpg'.format(i)), ncol=5)

                    if 'nuscenes' in labelset_name:
                        all_digits = np.unique(np.concatenate([np.unique(mapper[logits_pred].numpy()), np.unique(label)]))
                        labelset = list(NUSCENES_LABELS_16)
                        labelset[4] = 'construct. vehicle'
                        labelset[10] = 'road'
                        visualize_labels(list(all_digits), labelset, 
                            palette, os.path.join(save_folder, '{}_label.jpg'.format(i)), ncol=all_digits.shape[0])

                if eval_iou:
                    if mark_no_feature_to_unknown:
                        if "nuscenes" in labelset_name: # special case
                            masks.append(mask[inds_reverse][label_mask])
                        else:
                            masks.append(mask[inds_reverse])

                    if args.test_repeats==1:
                        # save directly the logits
                        preds.append(logits_pred)
                    else:
                        # only save the dot-product results, for repeat prediction
                        preds.append(pred.cpu())

                    gts.append(label.cpu())

            if eval_iou:
                gt = torch.cat(gts)
                pred = torch.cat(preds)
                
                if args.test_repeats==1 and True:
                    print(gt)
                    print(pred)
                    gt_ped_bicycle = (gt == 2) | (gt == 3)
                    pred_ped_bicycle = (pred == 2) | (pred == 3)

                    gt[gt_ped_bicycle] = 2
                    pred[pred_ped_bicycle] = 2

                pred_logit = pred
                if args.test_repeats>1:
                    pred_logit = pred.float().max(1)[1]

                if mapper is not None:
                    pred_logit = mapper[pred_logit]

                if mark_no_feature_to_unknown:
                    mask = torch.cat(masks)
                    pred_logit[~mask] = 256

                if args.test_repeats==1:
                    current_iou = metric.evaluate(pred_logit.numpy(),
                                                gt.numpy(),
                                                dataset=labelset_name,
                                                stdout=True)
                if args.test_repeats > 1:
                    store = pred + store
                    store_logit = store.float().max(1)[1]
                    if mapper is not None:
                        store_logit = mapper[store_logit]

                    if mark_no_feature_to_unknown:
                        store_logit[~mask] = 256
                    accumu_iou = metric.evaluate(store_logit.numpy(),
                                                gt.numpy(),
                                                stdout=True,
                                                dataset=labelset_name)

if __name__ == '__main__':
    main()

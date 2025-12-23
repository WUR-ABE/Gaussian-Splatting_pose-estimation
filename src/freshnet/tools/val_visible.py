# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
import numpy as np
import yaml
import shutil

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor

if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg


################# (@shlee) for save log
def printsave(*a,mode='a',log_path=None):
    with open(log_path, mode) as log_file:
        # print(*a)
        print(*a,file=log_file)


#import pdb;pdb.set_trace()
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--test_label_dir',
        type=str,
        help='Optional directory with labels. If not provided, default labels will be used. Required for evaluating at multiple visibility lower bounds.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()
    # os.environ['MKL_NUM_THREADS'] = '4'
    # os.environ['OMP_NUM_THREADS'] = '1'

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    val_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the val dataset is concatenated
    if isinstance(cfg.data.val, dict):
        cfg.data.val.test_mode = True
        if cfg.data.val_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
    elif isinstance(cfg.data.val, list):
        for ds_cfg in cfg.data.val:
            ds_cfg.test_mode = True
        if cfg.data.val_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.val:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    val_loader_cfg = {
        **val_dataloader_default_args,
        **cfg.data.get('val_dataloader', {})
    }

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(dataset, **val_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('val_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
        
    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            ### 원본
            # print(dataset.evaluate(outputs, show=args.show, out_dir=args.show_dir, ckpt_pth=args.checkpoint, **eval_kwargs)) ## (shlee) default: print(dataset.evaluate(outputs, show=args.show, out_dir=args.show_dir, **eval_kwargs))
            ### log 저장을 위한 코드
            # import pdb;pdb.set_trace()

            if args.test_label_dir is not None:
                # Define test label dir
                test_label_dir = args.test_label_dir

                # Loop the visibility values
                for visibility_lower_bound in cfg.test_visibility_values: 

                    for i in range(len(dataset.data_infos)):

                        idx = dataset.data_infos[i]['image']['image_idx']
                        label_filename = os.path.join(test_label_dir, "label", f'{idx:06d}.txt')
                        lines = [line.rstrip() for line in open(label_filename)]
                        annotations = {}

                        
                        if len(lines) != 0:
                            names = []
                            bboxes = []
                            locations = []
                            dimensions = []
                            classes = []
                            gt_boxes_upright_depth = []
                            ious = []
                            for line in lines:
                                data_line = line.split(' ')
                                data_line[1:] = [float(x) for x in data_line[1:]]
                                visibility = data_line[15]
                                # Check if the visibility is above the threshold
                                if visibility < visibility_lower_bound:
                                    continue
                                    
                                names.append(data_line[0])
                                classes.append(0) # Class 0 for apple
                                xmin = data_line[1]
                                ymin = data_line[3]
                                xmax = data_line[2]
                                ymax = data_line[4]
                                bboxes.append(np.array([xmin, ymin, xmax, ymax]))
                                centroid = np.array([data_line[5], data_line[6], data_line[7]]) # If increase for KITTI, * 10.0 
                                locations.append(centroid)
                                width = data_line[8]  / 2 # If increase for KITTI, * 10.0 
                                length = data_line[9]  / 2 # If increase for KITTI, * 10.0 
                                height = data_line[10]  / 2 # If increase for KITTI, * 10.0 
                                dimensions.append(np.array([length, width, height]))

                                size = np.array([float(data_line[9]), float(data_line[8]), float(data_line[10])]) # If increase for KITTI, * 10.0 
                                
                                roll = data_line[11]
                                pitch = data_line[12]
                                yaw = data_line[13]
                                
                                angle = np.array([roll, pitch, yaw])
                                box3d = np.concatenate([centroid, size, angle])
                                gt_boxes_upright_depth.append(box3d)
                                
                                ious.append(data_line[14])
                            
                            # Convert to numpy arrays and put in dict
                            annotations['gt_num'] = len(names)
                            annotations['name'] = np.array(names)
                            annotations['bbox'] = np.array(bboxes)
                            annotations['location'] = np.array(locations) * 10.0
                            annotations['dimensions'] = 2 * np.array(dimensions)
                            annotations['index'] = np.arange(len(lines), dtype=np.int32)
                            annotations['class'] = np.array(classes)
                            annotations['gt_boxes_upright_depth'] = np.array(gt_boxes_upright_depth) * 10.0
                            annotations['iou'] = np.array(ious)
                        else: 
                            annotations['gt_num'] = 0

                        # Replace the annotations in the dataset
                        dataset.data_infos[i]['annos'] = annotations

                    save_dir = os.path.dirname(args.out) ## (@ shlee)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_name = save_dir+"/{}_".format(cfg.data.val.ann_file.split("_")[-1].replace(".pkl",""))+"_".join(args.checkpoint.split("/")[-2:]).replace("pth","txt")
                    evaluation_dict = dataset.evaluate(outputs, show=args.show, out_dir=args.show_dir, ckpt_pth=args.checkpoint, **eval_kwargs)
                    evaluation_dict['visibility_lower_bound'] = visibility_lower_bound
                    printsave(evaluation_dict, log_path=save_name)
                    print("Log saved in:", save_name)

                # Copy the complete log to external dir
                train_visibility_lower_bound = yaml.safe_load(open("params.yaml"))["convert_randwijk_data_to_fresh"]["min_view_fraction"]
                real_data_fraction = yaml.safe_load(open("params.yaml"))["combine_real_and_splat_randwijk_data"]["real_fraction"]
                splat_data_fraction = yaml.safe_load(open("params.yaml"))["convert_splat_randwijk_data_to_fresh"]["real_fraction"]
                copied_save_name = f"data_disk/testing/testing_dataset/visibility_{int(train_visibility_lower_bound*100):02d}_realfraction_{int(real_data_fraction*1000):04d}_splatfraction_{int(splat_data_fraction*1000):04d}/" + os.path.basename(save_name)
                copied_save_dir = os.path.dirname(copied_save_name)
                if not os.path.exists(copied_save_dir):
                    os.makedirs(copied_save_dir)
                
                shutil.copyfile(save_name, copied_save_name)

            else:
                # Default evaluation
                save_dir = os.path.dirname(args.out)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_name = save_dir+"/{}_".format(cfg.data.val.ann_file.split("_")[-1].replace(".pkl",""))+"_".join(args.checkpoint.split("/")[-2:]).replace("pth","txt")
                printsave(dataset.evaluate(outputs, show=args.show, out_dir=args.show_dir, ckpt_pth=args.checkpoint, **eval_kwargs),log_path=save_name)
                print("Log saved in:", save_name)


if __name__ == '__main__':
    main()

voxel_size = .01
n_points = 100000

bbox_loss_weight = 1.0
angle_loss_weight = 1.0
score_thr = .01

optimizer_lr = 1e-4
optimizer_weight_decay = .0001
runner_max_epochs = 15

evaluation_interval = 20
data_train_times = 5

dvc_log_dir = 'dvclive/real_randwijk_freshnet_model/'
work_dir = '/home/gs-lfd/dvc-5d-apple-pose-estimation/models/real_randwijk_freshnet_model/'
data_root = '/home/gs-lfd/dvc-5d-apple-pose-estimation/data_disk/dvc_data/'
data_root_processed = '/home/gs-lfd/dvc-5d-apple-pose-estimation/data_disk/dvc_data/real_randwijk_papple_split/'

test_nms_pre = 100
test_iou_thr = 0.25

test_visibility_values = [
    0.0,  # 0 % visible
    0.001,# 0.1 % visible
    0.01, # 1 % visible
    0.05, # 5 % visible
    0.1,  # 10 % visible
    0.15, # 15 % visible
    0.2,  # 20 % visible
    0.25, # 25 % visible
    0.3,  # 30 % visible
    0.35, # 35 % visible
    0.4,  # 40 % visible
    0.45, # 45 % visible
    0.5,  # 50 % visible
]

model = dict(
    type='FRESHFFDetector',
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    backbone=dict(
        type='MinkResNet',
        in_channels=3,
        max_channels=128,
        depth=34,
        norm='batch'),
    neck=dict(
        type='TR3DNeck_angle',
        in_channels=(64, 128, 128, 128),
        out_channels=128),
    head=dict(
        type='TR3DHead_angle',
        in_channels=128,
        n_reg_outs=6,
        n_classes=1, 
        n_angle=3,
        voxel_size=voxel_size,
        assigner=dict(
            type='TR3DAssigner',
            top_pts_threshold=6,
            label2level=[1]),
        bbox_loss=dict(type='AxisAlignedIoULoss', mode='diou', reduction='none', loss_weight=bbox_loss_weight),
        angle_loss=dict(type='Angle3DLoss', mode='stem_vector', loss_weight=angle_loss_weight) ## mode: [ 'quat' 'stem_vector' ]
        ),
    voxel_size=voxel_size,
    train_cfg=dict(),
    test_cfg=dict(nms_pre=test_nms_pre, iou_thr=test_iou_thr, score_thr=score_thr))


optimizer = dict(type='Adam', lr=optimizer_lr, weight_decay=optimizer_weight_decay)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[8, 11, 14])
runner = dict(type='EpochBasedRunner', max_epochs=runner_max_epochs)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

checkpoint_config = dict(interval=3, max_keep_ckpts=6)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='DvcliveLoggerHook',
            report="md",
            path=dvc_log_dir,
            )
])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
evaluation=dict(interval=evaluation_interval)
find_unused_parameters=True


dataset_type = 'PappleDataset'
class_names = ('apple')
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        dataset_type=dataset_type,
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations3D'),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 504), (1333, 528), (1333, 552),
                   (1333, 576), (1333, 600)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='PointSample', num_points=n_points),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=.5,
        flip_ratio_bev_vertical=.0),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-.523599, .523599],
        scale_ratio_range=[.85, 1.15],
        translation_std=[.1, .1, .1],
        shift_height=False),
    # dict(type='NormalizePointsColor', color_mean=None),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        dataset_type=dataset_type,
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 600),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='Resize', multiscale_mode='value', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='PointSample', num_points=n_points),
            # dict(type='NormalizePointsColor', color_mean=None),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=data_train_times,
        dataset=dict(
            type=dataset_type,
            modality=dict(use_camera=True, use_lidar=True),
            data_root=data_root,
            ann_file=data_root_processed + 'papple_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=True,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        modality=dict(use_camera=True, use_lidar=True),
        data_root=data_root,
        ann_file=data_root_processed + 'papple_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        modality=dict(use_camera=True, use_lidar=True),
        data_root=data_root,
        ann_file=data_root_processed + 'papple_infos_test.pkl', 
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))

_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']

# model settings
model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type='mmcls.ShuffleNetV2',
        widen_factor=1.0,
        out_indices=(0, 1, 2),
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=dict(type='Pretrained', checkpoint='mmcls://shufflenet_v2')),
    neck=dict(
        type='NanoDetPAN',
        in_channels=[116, 232, 464],
        out_channels=96,
        start_level=0,
        num_outs=3),
    bbox_head=dict(
        type='NanoDetHead',
        num_classes=80,
        in_channels=96,
        stacked_convs=2,
        feat_channels=96,
        prior_box_scale=5,
        strides=[8, 16, 32],
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_dfl=dict(
            type='DistributionFocalLoss', loss_weight=0.25, reg_max=7)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=16),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# dataset settings
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        'data/coco/': 's3://openmmlab/datasets/detection/coco/',
        '.data/coco/': 's3://openmmlab/datasets/detection/coco/'
    }))
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(224, 224), (480, 480)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=48,
    workers_per_gpu=8,
    train=dict(
        _delete_=True,
        type='RepeatDataset',  # use RepeatDataset to speed up training
        times=10,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

evaluation = dict(by_epoch=True, interval=1, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.14, momentum=0.9, weight_decay=5.0e-4)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.0001,
    step=[24, 26, 27])
runner = dict(type='EpochBasedRunner', max_epochs=28)

find_unused_parameters = True

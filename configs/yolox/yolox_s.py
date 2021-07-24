_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='YOLOX',
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(type='YOLOXPAFPN',
              in_channels=[128, 256, 512],
              out_channels=128,
              csp_num_blocks=1
              ),
    bbox_head=dict(type='YOLOXHead',
                   num_classes=80,
                   in_channels=128,
                   feat_channels=128),
    # test
    test_cfg=dict(
        min_bbox_size=0,
        conf_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=1000)
)

img_norm_cfg = dict(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                    to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size=(640, 640), pad_val=114.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    test=dict(pipeline=test_pipeline))

# liner
batch_size = 2
basic_lr_per_img = 0.01 / 64.0

# optimizer
optimizer = dict(type='SGD', lr=batch_size * basic_lr_per_img, momentum=0.9, weight_decay=5e-4, nesterov=True,
                 paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='CosineAnnealingWithNoAugIter',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_iters=5,  # 5 epoch
    no_aug_epochs=15,
    min_lr_ratio=0.05)
runner = dict(type='EpochBasedRunner', max_epochs=300)
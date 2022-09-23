_base_ = './rtmdet_s_8xb32-300e_coco.py'

checkpoint = '/mnt/cache/lvchengqi/code/mmclassification/work_dirs/pretrain/cspnext-tiny/rsb-pretrain/epoch_600.pth'  # noqa

model = dict(
    backbone=dict(
        deepen_factor=0.167,
        widen_factor=0.375,
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(in_channels=[96, 192, 384], out_channels=96, num_csp_blocks=1),
    bbox_head=dict(in_channels=96, feat_channels=96))

train_pipeline = [
    dict(type='LoadImageFromFile',
         file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0),
    dict(type='RandomResize',
         scale=(1280, 1280),
         ratio_range=(0.5, 2.0),
         keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='CachedMixUp',
         img_scale=(640, 640),
         ratio_range=(1.0, 1.0),
         max_cached_images=20,
         pad_val=(114, 114, 114),
         prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

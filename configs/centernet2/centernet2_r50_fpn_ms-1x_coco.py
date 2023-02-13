_base_ = '../common/ms-90k_coco.py'

custom_imports = dict(
    imports=['projects.Detic.detic'], allow_failed_imports=False)

model = dict(
    type='CascadeRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        add_extra_convs='on_output',
        num_outs=5,
        init_cfg=dict(type='Caffe2Xavier', layer='Conv2d'),
        relu_before_extra_convs=True),
    rpn_head=dict(
        type='CenterNetRPNHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        conv_bias=True,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        loss_cls=dict(
            type='GaussianFocalLoss',
            pos_weight=0.5,
            neg_weight=0.5,
            loss_weight=1.0,
            fp_ignore_thr=0.85),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
    ),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=7,
                sampling_ratio=0,
                use_torchvision=True),
            out_channels=256,
            featmap_strides=[8, 16, 32],
            # approximately equal to
            # canonical_box_size=224, canonical_level=4 in D2
            finest_scale=112),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=4000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.9),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.8,
                    neg_iou_thr=0.8,
                    min_pos_iou=0.8,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            score_thr=0.0001,
            nms_pre=1000,
            max_per_img=256,
            nms=dict(type='nms', iou_threshold=0.9),
            min_bbox_size=0),
        rcnn=dict(
            mul_proposal_scores=True,
            score_thr=0.02,
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=100)))

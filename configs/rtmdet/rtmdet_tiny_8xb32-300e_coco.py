_base_ = './rtmdet_s_8xb32-300e_coco.py'

model = dict(
    backbone=dict(deepen_factor=0.167, widen_factor=0.375),
    neck=dict(in_channels=[96, 192, 384], out_channels=96, num_csp_blocks=1),
    bbox_head=dict(in_channels=96, feat_channels=96))

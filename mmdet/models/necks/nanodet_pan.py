import torch.nn.functional as F
from mmcv.runner import auto_fp16

from ..builder import HEADS
from .fpn import FPN


@HEADS.register_module()
class NanoDetPAN(FPN):
    """A lite version of Path Aggregation Network used in NanoDet.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): Whether to add conv layers on top of the
            original feature maps. Default: False.
        extra_convs_on_inputs (bool): Whether to apply extra conv on
            the original feature from the backbone. Default: False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 upsample_cfg=dict(mode='bilinear', scale_factor=2),
                 downsample_cfg=dict(mode='bilinear', scale_factor=0.5),
                 **kwargs):
        super(NanoDetPAN, self).__init__(
            in_channels,
            out_channels,
            num_outs,
            upsample_cfg=upsample_cfg,
            **kwargs)
        delattr(self, 'fpn_convs')
        self.downsample_cfg = downsample_cfg.copy()

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        inter_outs = [laterals[i] for i in range(used_backbone_levels)]

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            if 'scale_factor' in self.downsample_cfg:
                inter_outs[i + 1] += F.interpolate(inter_outs[i],
                                                   **self.downsample_cfg)
            else:
                prev_shape = inter_outs[i + 1].shape[2:]
                inter_outs[i + 1] += F.interpolate(
                    inter_outs[i], size=prev_shape, **self.downsample_cfg)

        outs = [inter_outs[0]]
        outs.extend([inter_outs[i] for i in range(1, used_backbone_levels)])

        return tuple(outs)

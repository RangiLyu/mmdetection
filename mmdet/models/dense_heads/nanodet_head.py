import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      bias_init_with_prob, constant_init, is_norm, normal_init)
from mmcv.runner import force_fp32

from mmdet.core import (bbox2distance, bbox_overlaps, build_assigner,
                        build_sampler, distance2bbox, images_to_levels,
                        multi_apply, multiclass_nms, reduce_mean)
from ..builder import HEADS, LOSSES
from .anchor_free_head import AnchorFreeHead
from .gfl_head import Integral


@HEADS.register_module()
class NanoDetHead(AnchorFreeHead):
    r"""Implementation of  `NanoDet: Super fast and lightweight anchor-free
    object detection model <https://github.com/RangiLyu/nanodet>`_

    NanoDet is a FCOS-style one-stage anchor-free object detection model
    which using ATSS for target sampling and using Generalized Focal Loss
    for classification and box regression.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        prior_box_scale (int): Scale of prior box for ATSS target assign.
            Default: 4.
        use_depthwise (bool): Whether to use depthwise separable conv in
            heads. Default: None.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_dfl (dict): Config of DistributionFocalLoss. If None, use FCOS
            box regression.
        act_cfg (dict): dictionary to construct and config activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1)
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN').
        init_cfg (dict or list[dict], optional): Initialization config dict."""

    def __init__(self,
                 num_classes,
                 in_channels,
                 prior_box_scale=4,
                 use_depthwise=True,
                 loss_cls=dict(
                     type='mmdet.QualityFocalLoss',
                     use_sigmoid=True,
                     beta=2.0,
                     loss_weight=1.0),
                 loss_bbox=dict(type='mmdet.GIoULoss', loss_weight=1.0),
                 loss_dfl=dict(
                     type='mmdet.DistributionFocalLoss',
                     reg_max=8,
                     loss_weight=0.25),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 norm_cfg=dict(type='BN'),
                 init_cfg=None,
                 **kwargs):

        self.prior_box_scale = prior_box_scale
        self.use_depthwise = use_depthwise
        if loss_dfl:
            self.use_dfl_loss = True
            self.reg_max = loss_dfl.pop('reg_max')
            self.reg_out_channels = (self.reg_max + 1) * 4
        else:
            self.use_dfl_loss = False
            self.reg_out_channels = 4
        self.act_cfg = act_cfg
        super(NanoDetHead, self).__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        if self.use_dfl_loss:
            self.integral = Integral(self.reg_max)
            self.loss_dfl = LOSSES.build(loss_dfl)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.conv_heads = nn.ModuleList()
        for _ in self.strides:
            self.conv_heads.append(self._init_conv_head())

    def _init_conv_head(self):
        """Initialize detection head conv layers of one level's head."""
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule
        conv_head = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            conv_head.append(
                conv(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.conv_bias))

        conv_head.append(
            nn.Conv2d(self.feat_channels,
                      self.cls_out_channels + self.reg_out_channels, 1))
        return nn.Sequential(*conv_head)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)

        # Use prior in model initialization to improve stability
        for conv_head in self.conv_heads:
            bias = conv_head[-1].bias
            nn.init.constant_(bias.data, bias_init_with_prob(0.01))

            if not self.use_dfl_loss:
                nn.init.constant_(bias.data[self.cls_out_channels:],
                                  self.prior_box_scale)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually contain classification scores and bbox predictions.
                cls_scores (list[Tensor]): Box scores for each scale level,
                    each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * 4.
        """
        return multi_apply(self.forward_single, feats, self.conv_heads)

    def forward_single(self, x, conv_head):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            conv_head (nn.Module): Head of each stages.

        Returns:
            tuple: Scores for each class, bbox predictions.
        """
        x = conv_head(x)
        cls_score, bbox_pred = torch.split(
            x, [self.cls_out_channels, self.reg_out_channels], dim=1)
        if not self.use_dfl_loss:
            bbox_pred = F.relu(bbox_pred)
        return cls_score, bbox_pred

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level has
                shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box regression predictions for each
                scale level with shape (N, reg_out_channels, H, W),
                reg_out_channels is 4 when not using DFL otherwise is
                4 * (reg_max + 1).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        device = cls_scores[0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        multi_level_points = self.get_points(
            featmap_sizes, bbox_preds[0].dtype, device, flatten=True)

        cls_reg_targets = self.get_targets(
            multi_level_points,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels)
        if cls_reg_targets is None:
            return None

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, losses_dfl, avg_factor = multi_apply(
            self.loss_single,
            multi_level_points,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            self.strides,
            num_total_samples=num_total_samples)

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))
        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl)

    def loss_single(self, points, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, stride, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            points (Tensor): Box reference for each scale level with shape
                (N, num_total_points, 4).
            cls_score (Tensor): Box scores for each scale level has shape
                (N, num_classes, H, W).
            bbox_pred (Tensor): Box regression predictions for each scale level
                with shape (N, reg_out_channels, H, W), reg_out_channels is 4
                when not using DFL otherwise is 4 * (reg_max + 1).
            labels (Tensor): Labels of each points with shape
                (N, num_total_points).
            label_weights (Tensor): Label weights of each point with shape
                (N, num_total_points)
            bbox_targets (Tensor): BBox regression targets of each point with
                shape (N, num_total_points, 4).
            stride (tuple): Stride in this scale level.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        y, x = points
        centers = torch.cat([torch.stack(
            (x, y), dim=-1)] * cls_score.size()[0])
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, self.reg_out_channels)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_centers = centers[pos_inds] / stride

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]

            pos_bbox_pred_corners = self.integral(pos_bbox_pred) \
                if self.use_dfl_loss else pos_bbox_pred
            pos_decode_bbox_pred = distance2bbox(pos_centers,
                                                 pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride
            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)

            if self.use_dfl_loss:
                pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
                target_corners = bbox2distance(pos_centers,
                                               pos_decode_bbox_targets,
                                               self.reg_max).reshape(-1)
                loss_dfl = self.loss_dfl(
                    pred_corners,
                    target_corners,
                    weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                    avg_factor=4.0)
            else:
                loss_dfl = bbox_pred.sum() * 0
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)

        # cls (qfl) loss
        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples)

        return loss_cls, loss_bbox, loss_dfl, weight_targets.sum()

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_classes, H, W)
            bbox_preds (list[Tensor]): Box regression predictions for each
                scale level with shape (N, reg_out_channels, H, W).
                reg_out_channels is 4 when not using DFL otherwise is
                4 * (reg_max + 1).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_points = self.get_points(
            featmap_sizes,
            bbox_preds[0].dtype,
            bbox_preds[0].device,
            flatten=True)

        mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]

        img_shapes = [
            img_metas[i]['img_shape'] for i in range(cls_scores[0].shape[0])
        ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]

        result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds,
                                       mlvl_points, img_shapes, scale_factors,
                                       cfg, rescale)

        return result_list

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    mlvl_points,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box regression predictions for each
                scale level with shape (N, reg_out_channels, H, W).
                reg_out_channels is 4 when not using DFL otherwise is
                4 * (reg_max + 1).
            img_shapes (list[tuple[int]]): Shape of the input image,
                list[(height, width, 3)].
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, stride, points in zip(
                cls_scores, bbox_preds, self.strides, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            y, x = points
            centers = torch.stack((x, y), dim=-1)
            centers = centers.expand(batch_size, -1, 2)
            scores = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            if self.use_dfl_loss:
                bbox_pred = self.integral(bbox_pred)
            bbox_pred = bbox_pred * stride
            bbox_pred = bbox_pred.reshape(batch_size, -1, 4)

            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                max_scores, _ = scores.max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                centers = centers[batch_inds, topk_inds, :]
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]

            bboxes = distance2bbox(centers, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)

        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = batch_mlvl_scores.new_zeros(batch_size,
                                              batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes,
                                                  batch_mlvl_scores):
                det_bbox, det_label = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                     cfg.score_thr, cfg.nms,
                                                     cfg.max_per_img)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)
            ]
        return det_results

    def get_targets(self,
                    multi_level_points,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None):
        """Compute regression, classification targets for points in multiple
        images.

        Args:
            multi_level_points (list[Tensor]): Points of each fpn level,
                each has shape (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                mlvl_labels (list[Tensor]): Labels of each level.
                mlvl_label_weights (list[Tensor]): Labels weight of each level.
                mlvl_bbox_targets (list[Tensor]): BBox targets of each level.
                mlvl_bbox_weights (list[Tensor]): BBox weights of each level.
                num_total_pos (Tensor): Total number of positive targets.
                num_total_neg (Tensor): Total number of negative targets.
        """
        num_imgs = len(img_metas)
        num_level_points = [points[0].size(0) for points in multi_level_points]
        multi_level_points_list = [multi_level_points] * num_imgs
        num_level_points_list = [num_level_points] * num_imgs
        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        # target assign on all images, get list of tensors
        # list length = batch size
        # tensor first dim = num of all points
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single,
                                      multi_level_points_list,
                                      num_level_points_list, gt_bboxes_list,
                                      gt_bboxes_ignore_list, gt_labels_list)
        # no valid points
        if any([labels is None for labels in all_labels]):
            return None
        # sampled points of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # merge list of targets tensors into one batch
        # then split to multi levels
        mlvl_labels = images_to_levels(all_labels, num_level_points)
        mlvl_label_weights = images_to_levels(all_label_weights,
                                              num_level_points)
        mlvl_bbox_targets = images_to_levels(all_bbox_targets,
                                             num_level_points)
        mlvl_bbox_weights = images_to_levels(all_bbox_weights,
                                             num_level_points)
        return (mlvl_labels, mlvl_label_weights, mlvl_bbox_targets,
                mlvl_bbox_weights, num_total_pos, num_total_neg)

    def _get_target_single(self, mlvl_points, num_level_points, gt_bboxes,
                           gt_bboxes_ignore, gt_labels):
        """Compute regression, classification targets for points in a single
        image.

        Args:
            mlvl_points (List[Tensor]): Multi-level points of the image.
            num_level_points (List[int]): Number of points of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).

        Returns:
            tuple: N is the number of total points in the image.
                labels (Tensor): Labels of all points in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all points in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all points in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all points in the
                    image with shape (N, 4).
                pos_inds (Tensor): Indices of positive points with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative points with shape
                    (num_neg,).
        """
        all_priors = []
        for points, stride in zip(mlvl_points, self.strides):
            y, x = points
            cell_size = self.prior_box_scale * stride
            prior_boxes = [
                x - 0.5 * cell_size, y - 0.5 * cell_size, x + 0.5 * cell_size,
                y + 0.5 * cell_size
            ]
            prior_boxes = torch.stack(prior_boxes, dim=-1)
            all_priors.append(prior_boxes)
        all_priors = torch.cat(all_priors)
        assign_result = self.assigner.assign(all_priors, num_level_points,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, all_priors,
                                              gt_bboxes)

        num_points = all_priors.shape[0]
        bbox_targets = torch.zeros_like(all_priors)
        bbox_weights = torch.zeros_like(all_priors)
        labels = all_priors.new_full((num_points, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = all_priors.new_zeros(num_points, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]

            label_weights[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points of a single scale level."""
        h, w = featmap_size
        # First create Range with the default dtype, than convert to
        # target `dtype` for onnx exporting.
        x_range = (torch.arange(w, dtype=dtype, device=device) + 0.5) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device) + 0.5) * stride
        y, x = torch.meshgrid(y_range, x_range)
        if flatten:
            y = y.flatten()
            x = x.flatten()
        return y, x

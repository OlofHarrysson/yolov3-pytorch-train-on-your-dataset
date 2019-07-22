import torch
import torch.nn as nn
import torch.nn.functional as F
from .dataclasses import BboxTarget, TargetsHelper
import numpy as np
from torch import FloatTensor
from utils.utils import bbox_ious


class YoloHead():
  def __init__(self, anchors, device, config, n_classes):
    super(YoloHead, self).__init__()
    self.device = device 
    self.config = config
    self.anchors = anchors
    self.n_classes = n_classes

    self.anchor_xs = FloatTensor([anch[0] for anch in anchors])
    self.anchor_ys = FloatTensor([anch[1] for anch in anchors])

  def predict(self, fmap, image_size):
    # TODO: Dont predict before you sigmoid the objclass confs
    fmap = fmap.cpu() # No speedup on GPU from here on
    batch_size = fmap.size(0)
    n_grids = fmap.size(2)
    n_anchors = len(self.anchors)
    stride = image_size / n_grids
    anchor_xs = self.anchor_xs
    anchor_ys = self.anchor_ys

    x, y, width, height, obj_conf, class_confs = self.slice_output(fmap)
    # obj_conf = torch.sigmoid(obj_conf)
    # class_confs = torch.sigmoid(class_confs)

    # Offset x & y so they end up in different cells
    grid = np.arange(n_grids)
    x_offset, y_offset = np.meshgrid(grid, grid)
    x += FloatTensor(x_offset)
    y += FloatTensor(y_offset)

    # Trick to get around broadcasting rules. Same as w * a
    multiply = lambda w, a: (w.transpose(1, 3) * a).transpose(1, 3)

    pred_boxes = FloatTensor(batch_size, n_anchors, n_grids, n_grids, 4)
    pred_boxes[..., 0] = x * stride
    pred_boxes[..., 1] = y * stride
    pred_boxes[..., 2] = multiply(torch.exp(width), anchor_xs)
    pred_boxes[..., 3] = multiply(torch.exp(height), anchor_ys)

    output = torch.cat(
      (pred_boxes.view(batch_size, -1, 4),
        obj_conf.view(batch_size, -1, 1),
        class_confs.view(batch_size, -1, self.n_classes)
      ), -1,)
    return output

  def slice_output(self, fmap):
    batch_size = fmap.size(0)
    n_grids = fmap.size(2)
    n_anchors = len(self.anchors)
    bbox_attrs = 5 + self.n_classes

    preds = fmap.view(batch_size, n_anchors, bbox_attrs, n_grids, n_grids)
    preds = preds.permute(0, 1, 3, 4, 2).contiguous()

    # TODO: Can I skip permute by slicing in all the other dimensions? Can make lambda function
    x = torch.sigmoid(preds[..., 0])
    y = torch.sigmoid(preds[..., 1])
    width = preds[..., 2]
    height = preds[..., 3]
    obj_conf = torch.sigmoid(preds[..., 4])
    class_confs = torch.sigmoid(preds[..., 5:])

    return x, y, width, height, obj_conf, class_confs

  def calc_loss(self, fmap, targets_raw, image_size):
    ''' Loss for one detection layer '''
    fmap = fmap.to(self.device)
    sliced_fmap = self.slice_output(fmap)
    x, y, w, h, obj_conf, class_confs = sliced_fmap
    targ = self.build_targets(targets_raw, fmap, image_size, x.size(), class_confs.size())

    # Mask for indices with gt objects and thresh IoU
    mask = targ.has_obj_mask

    cfg = self.config
    # x, y, width, height loss
    loss_x = cfg.lambda_x * F.mse_loss(x[mask], targ.x[mask])
    loss_y = cfg.lambda_y * F.mse_loss(y[mask], targ.y[mask])
    loss_w = cfg.lambda_w * F.mse_loss(w[mask], targ.width[mask])
    loss_h = cfg.lambda_h * F.mse_loss(h[mask], targ.height[mask])

    # Loss for cells with gt
    bce_loss = F.binary_cross_entropy
    obj_pred = obj_conf[targ.obj_conf]
    obj_loss_gt = bce_loss(obj_pred, torch.ones_like(obj_pred))
    obj_loss_gt *= cfg.lambda_obj_gt

    # Loss for cells without gt
    no_obj = obj_conf[~targ.has_obj_mask]
    obj_loss_no_gt = bce_loss(no_obj, torch.zeros_like(no_obj))
    obj_loss_no_gt *= cfg.lambda_obj_no_gt

    # Class loss
    ce_loss = F.cross_entropy # TODO: Change to bce
    cls_tar = torch.argmax(targ.class_conf[mask], 1)
    loss_cls = ce_loss(class_confs[mask], cls_tar)
    loss_cls *= cfg.lambda_class

    return loss_x, loss_y, loss_w, loss_h, obj_loss_gt, obj_loss_no_gt, loss_cls

  def build_targets(self, targets_raw, fmap, image_size, t_shape, cls_shape):
    # TODO: need two masks, one for GT and one for gt that cares about ignore thresh. Then I should not just set best box but all the priors for the GT one
    batch_size = fmap.size(0)
    n_grids = fmap.size(2)
    n_anchors = len(self.anchors)
    stride = image_size / n_grids
    anchors = self.anchors
  
    anchs = np.append(np.zeros((n_anchors, 2)), anchors, axis=1)
    anchor_shapes = anchs / stride

    t_x = np.zeros(t_shape)
    t_y = np.zeros(t_shape)
    t_width = np.zeros(t_shape)
    t_height = np.zeros(t_shape)
    t_obj_conf = np.ones(t_shape)
    has_obj_mask = np.zeros(t_shape)
    t_class_conf = np.zeros(cls_shape)

    for batch in range(batch_size):
      target_batch = targets_raw[batch]
      for target in target_batch:
        target = BboxTarget(target) # Convenience class
        grid_x = target.x * n_grids
        grid_y = target.y * n_grids
        w = target.width * n_grids
        h = target.height * n_grids

        # The grid cell the target lies in
        x_cell = int(grid_x)
        y_cell = int(grid_y)

        # Find the best matching anchor box
        gt_box = np.array([[0, 0, w, h]])
        ious = bbox_ious(gt_box, anchor_shapes)
        best_box = np.argmax(ious)

        # Create X & Y targets
        t_x[batch, best_box, y_cell, x_cell] = grid_x - x_cell
        t_y[batch, best_box, y_cell, x_cell] = grid_y - y_cell

        # Create width & height targets
        w = target.width * image_size / anchors[best_box][0]
        h = target.height * image_size / anchors[best_box][1]
        t_width[batch, best_box, y_cell, x_cell] = np.log(w + 1e-16)
        t_height[batch, best_box, y_cell, x_cell] = np.log(h + 1e-16)
       
        # Create confidence targets
        ignore_thres = 0.5
        t_obj_conf[batch, ious > ignore_thres, y_cell, x_cell] = 0
        t_obj_conf[batch, best_box, y_cell, x_cell] = 1
        t_class_conf[batch, best_box, y_cell, x_cell, target.class_int] = 1

        # Masks
        has_obj_mask[batch, :, y_cell, x_cell] = 1

    # Convenience class
    return TargetsHelper(t_x, t_y, t_width, t_height, t_obj_conf, t_class_conf, has_obj_mask, self.device)
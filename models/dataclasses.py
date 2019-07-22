import numpy as np
import torch

class BboxTarget():
  def __init__(self, bbox):
    self.bbox = bbox

  @property
  def class_int(self):
    return self.bbox[0].astype(np.int)

  @property
  def x(self):
    return self.bbox[1]

  @property
  def y(self):
    return self.bbox[2]

  @property
  def width(self):
    return self.bbox[3]

  @property
  def height(self):
    return self.bbox[4]

  def __str__(self):
    return str(self.bbox)
    
class TargetsHelper():
  def __init__(self, t_x, t_y, t_width, t_height, t_obj_conf, t_class_conf, has_obj_mask, device):
    dev = device
    self.x = torch.from_numpy(t_x).float().to(dev)
    self.y = torch.from_numpy(t_y).float().to(dev)
    self.width = torch.from_numpy(t_width).float().to(dev)
    self.height = torch.from_numpy(t_height).float().to(dev)
    self.obj_conf = torch.from_numpy(t_obj_conf).byte().to(dev)
    self.class_conf = torch.from_numpy(t_class_conf).long().to(dev)
    self.has_obj_mask = torch.from_numpy(has_obj_mask).byte().to(dev)

class DatasetHelper():
  def __init__(self, im_path, target_bboxes, unpad):
    self.im_path = im_path
    self.target_bboxes = target_bboxes
    self.unpad = unpad
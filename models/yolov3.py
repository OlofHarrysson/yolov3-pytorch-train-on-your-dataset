import torch
import torch.nn as nn
from .modules import FeatureExtractor
from .yolo_head import YoloHead
from collections import defaultdict

# TODO: Better way to acces certain layers for weight loading, freeze params?
class Yolov3(nn.Module):
  def __init__(self, n_classes, config):
    super().__init__()
    self.name = self.__class__.__name__
    self.use_gpu = config.use_gpu
    self.device = 'cuda' if self.use_gpu else 'cpu'
    self.init_modules(n_classes, config)
    self.n_classes = n_classes

  def forward(self, x):
    err_size = "Yolov3 can't handle this image size"
    assert x.size(2) % 32 == 0, err_size
    x = x.to(self.device)
    return self.feature_extractor(x)

  def predict(self, feature_maps, image_size):
    outputs = []
    for detector, fmap in zip(self.detectors, feature_maps):
      prediction = detector.predict(fmap, image_size)
      outputs.append(prediction)

    return torch.cat(outputs, 1)

  def calc_loss(self, fmaps, targets, image_size):
    st = lambda: torch.zeros([1], dtype=torch.float32).to(self.device)
    dict_losses = defaultdict(st)
    loss_names = ['x', 'y', 'w', 'h', 'obj_gt', 'obj_no_gt', 'cls']
    
    for detector, fmap in zip(self.detectors, fmaps):
      losses = detector.calc_loss(fmap, targets, image_size)
      for name, loss in zip(loss_names, losses):
        dict_losses[name] = torch.add(dict_losses[name], loss)

    return dict_losses

  def load_weights(self, path, load_last=True):
    ''' Loads a weight file from path. The load_last flag controls if the last detection layers controlling the amount of classes should load '''
    print('Loading weights from {}'.format(path))
    # Get all the layers that shouldn't be loaded
    dont_load = []
    if load_last == False:
      det_w = self.state_dict()
      for name in det_w:
        if 'yolo_head' in name:
          dont_load.append(name)

    # Load and filter out the weights not to load
    weights = torch.load(path, map_location='cpu')
    do_load = {k: v for k, v in weights.items() if k not in dont_load}
    try:
      self.load_state_dict(do_load, strict=False)
    except RuntimeError:
      err_str = "Weights file is funky. Make sure your weights file matches this model's number of classes which is {}".format(self.n_classes)
      raise RuntimeError(err_str)

  def save_weights(self, path):
    print("Saving Weights @ " + path)
    torch.save(self.state_dict(), path)

  def freeze_weights(self, layers):
    self.feature_extractor.freeze_weights(layers)

  def init_anchors(self):
    # TODO: Move to file?
    return [[(116,90), (156,198), (373,326)],
      [(30,61), (62,45), (59,119)],
      [(10,13), (16,30), (33,23)]]

  def init_modules(self, n_classes, cfg):
    self.anchors = self.init_anchors()
    self.feature_extractor = FeatureExtractor(n_classes)

    self.detectors = []
    add_head = lambda x: self.detectors.append(x)
    for anchors in self.anchors:
      add_head(YoloHead(anchors, self.device, cfg, n_classes))
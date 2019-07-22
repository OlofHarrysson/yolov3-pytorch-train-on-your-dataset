import torch
import torch.nn as nn
import torch.nn.functional as F
from .helper_modules import ConvSeries, conv_block, Residual

class FeatureExtractor(nn.Module):
  def __init__(self, n_classes):
    super().__init__()
    self.layers = nn.ModuleDict()
    self.n_head_filters = 3 * (5 + n_classes)

    def n_out_filters(n):
      ''' Helper to get output filters from modules. n=index '''
      get_module_at_index = lambda i : list(self.layers.values())[i]
      return get_module_at_index(n).n_output_filters

    self.darknet53 = Darknet53()
    self.init_lowres_branch(n_out_filters, self.layers)
    self.init_midres_branch(n_out_filters, self.layers)
    self.init_highres_branch(n_out_filters, self.layers)

  def init_lowres_branch(self, n_out_filters, layers):
    # Series of conv layers, (n_out_channels, kernel_size)
    shapes = [(512,1), (1024,3), (512,1), (1024,3), (512,1)]
    prev_filters = self.darknet53.n_out_filters(-1)
    layers['branch1_convs'] = ConvSeries(shapes, prev_filters)

    # First detection head
    shapes = [(1024, 3), (self.n_head_filters, 1)]
    layers['yolo_head1'] = DetectionHead(shapes, n_out_filters(-1))

  def init_midres_branch(self, n_out_filters, layers):
    dnet53 = self.darknet53
    layers['branch2_start'] = ConvSeries([(256, 1)], n_out_filters(-2))

    # First upsample
    layers['up1'] = UpsampleRoute()

    # Series of conv layers, (n_out_channels, kernel_size)
    shapes = [(256, 1), (512, 3), (256, 1), (512, 3), (256, 1)]
    prev_filters = n_out_filters(-2) + dnet53.n_out_filters(-2)
    layers['branch2_convs'] = ConvSeries(shapes, prev_filters)
    
    # Second detection head
    shapes = [(512, 3), (self.n_head_filters, 1)]
    layers['yolo_head2'] = DetectionHead(shapes, n_out_filters(-1))

  def init_highres_branch(self, n_out_filters, layers):
    dnet53 = self.darknet53
    layers['branch3_start'] = ConvSeries([(128, 1)], n_out_filters(-2))

    # Second upsample
    layers['up2'] = UpsampleRoute()

    # Series of conv layers, (n_out_channels, kernel_size)
    shapes = [(128, 1), (256, 3), (128, 1), (256, 3), (128, 1)]
    prev_filters = n_out_filters(-2) + dnet53.n_out_filters(-3)
    layers['branch3_convs'] = ConvSeries(shapes, prev_filters)

    # Third detection head
    shapes = [(256, 3), (self.n_head_filters, 1)]
    layers['yolo_head3'] = DetectionHead(shapes, n_out_filters(-1))

  def forward(self, x):
    dn53_fmaps = self.darknet53(x)
    x = dn53_fmaps.pop()

    features = []
    for module in self.layers.values():
      if isinstance(module, DetectionHead):
        features.append(module(x))
      elif isinstance(module, UpsampleRoute):
        route = dn53_fmaps.pop()
        x = module(x, route)
      else:
        x = module(x)

    return features

  def freeze_weights(self, layers):
    print("Freezing {} weights...".format(layers))
    def freeze(weights):
      for w in weights:
        w.requires_grad = False

    def unfreeze(weights):
      for w in weights:
        w.requires_grad = True

    if layers == 'all_but_last':
      # Freeze all and unfreeze heads
      freeze(self.parameters())
      get_heads = lambda mod: isinstance(mod, DetectionHead)
      heads = filter(get_heads, self.modules())
      params = map(lambda x: x.parameters(), heads)
      list(map(unfreeze, params))
    elif layers == 'dn53':
      freeze(self.darknet53.parameters())
    elif layers == 'all':
      freeze(self.parameters())
    else:
      raise NotImplementedError("Can't freeze these layers -> {}".format(layers))


class Darknet53(nn.Module):
  def __init__(self):
    super().__init__()
    self.mods = self.init_modules()
    
  def init_modules(self):
    modules = nn.ModuleList()

    inp_channels = 3
    modules.append(conv_block(inp_channels, 32, 3, pad=1))

    # Residual shapes. (n_input_filters, n_blocks)
    res_shapes = [(32,1), (64,2), (128,8), (256,8), (512,4)]
    prev_filters = 32
    for n_filters, n_res_blocks in res_shapes:
      mod = Residual(n_filters, n_res_blocks, prev_filters)
      prev_filters = mod.n_output_filters
      modules.append(mod)

    self.n_output_filters = prev_filters
    return modules

  def forward(self, x):
    fmaps = []
    for i, module in enumerate(self.mods):
      x = module(x)
      if i > 2:
        # Save routing feature maps
        fmaps.append(x)

    return fmaps

  def n_out_filters(self, i):
    return self.mods[i].n_output_filters

class DetectionHead(nn.Module):
  def __init__(self, shapes, prev_filters):
    super().__init__()
    shapes = iter(shapes)
    n_filters, kernel = next(shapes)
    c1 = conv_block(prev_filters, n_filters, kernel, pad=1)

    # Last layer has no batch norm, is linear with bias
    prev_filters = n_filters
    n_filters, kernel = next(shapes)
    c2 = nn.Conv2d(prev_filters, n_filters, 1)
    self.mods = nn.Sequential(*[c1, c2])

  def forward(self, x):
    return self.mods(x)

class UpsampleRoute(nn.Module):
  ''' Upsamples x and concats it with a feature map routed from earlier in the network '''
  def __init__(self):
    super().__init__()

  def forward(self, x, route):
    # Upsample
    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

    # Concat
    return torch.cat((x, route), dim=1)
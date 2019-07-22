import torch.nn as nn

class conv_block(nn.Module):
  ''' Conv2d + Batchnorm + LeakyRelu '''
  def __init__(self, n_in, n_out, kernel, stride=1, pad=0, bias=False):
    super().__init__()
    b = []
    b.append(nn.Conv2d(n_in, n_out, kernel, stride=stride, padding=pad, bias=bias))
    b.append(nn.BatchNorm2d(n_out))
    b.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
    self.mods = nn.Sequential(*b)

  def forward(self, x):
    return self.mods(x)


class ConvSeries(nn.Module):
  def __init__(self, shapes, prev_filters):
    super().__init__()
    layers = []
    for filters, kernel in shapes:
      pad = 1 if kernel == 3 else 0
      block = conv_block(prev_filters, filters, kernel, pad=pad)
      layers.append(block)
      prev_filters = filters

    self.mods = nn.Sequential(*layers)
    self.n_output_filters = prev_filters

  def forward(self, x):
    return self.mods(x)


class Residual(nn.Module):
  ''' Residual block. Starts with a downsample and then a bunch of residual connections a la YOLO style (specific pattern) '''
  def __init__(self, n_filters, n_blocks, prev_filters):
    super().__init__()
    self.down_sample = conv_block(prev_filters, n_filters*2, 3, stride=2, pad=1)

    b = []
    for i in range(1, n_blocks + 1):
      b.append(conv_block(n_filters*2, n_filters, 1))
      b.append(conv_block(n_filters, n_filters*2, 3, pad=1))
      b.append(ShortcutPlaceholder())

    self.mods = nn.ModuleList(b)
    self.n_output_filters = n_filters * 2


  def forward(self, x):
    x = self.down_sample(x)
    down_sample = x

    for module in self.mods:
      if isinstance(module, ShortcutPlaceholder):
        x = x + down_sample
        down_sample = x
      else:
        x = module(x)

    return x


class ShortcutPlaceholder(nn.Module):
  ''' AKA skip connection '''
  def __init__(self):
    super().__init__()
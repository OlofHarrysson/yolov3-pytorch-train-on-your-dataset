import wget, hashlib, torch, sys
from pathlib import Path

# Needed to import from project root
main_dir = Path.cwd().parent.parent
sys.path.append(str(main_dir))
from models.yolov3 import Yolov3
import numpy as np
from config.base_config import NoCuda
import torch.nn as nn
from models.helper_modules import conv_block

# info about weight structure
# https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe

def convert_weights(weights_path, model):
  def copy_weights(ptr, target_weight, num_ele):
    p = torch.from_numpy(weights[ptr : ptr + num_ele])
    target_weight.data.copy_(p.view_as(target_weight))
    return num_ele

  def load_weights(module):
    global ptr

    # Layers with conv2d + batch norm
    if type(module) == conv_block:
      conv_layer, bn_layer = module.mods[0], module.mods[1]

      # Copy batch norm + conv weights
      n_bn = bn_layer.bias.numel() # Batch norm size
      ptr += copy_weights(ptr, bn_layer.bias, n_bn)
      ptr += copy_weights(ptr, bn_layer.weight, n_bn)
      ptr += copy_weights(ptr, bn_layer.running_mean, n_bn)
      ptr += copy_weights(ptr, bn_layer.running_var, n_bn)

      n_conv = conv_layer.weight.numel() # Conv2d weight size
      ptr += copy_weights(ptr, conv_layer.weight, n_conv)

    # Detection heads without batch norm
    if type(module) == nn.Conv2d:
      try:
        n_bias = module.bias.numel()
        ptr += copy_weights(ptr, module.bias, n_bias)

        n_w = module.weight.numel()
        ptr += copy_weights(ptr, module.weight, n_w)

      except AttributeError:
        pass

  with open(weights_path, "rb") as fp:
    headers = np.fromfile(fp, dtype=np.int32, count=5)
    weights = np.fromfile(fp, dtype=np.float32)
  weight_size = len(weights)

  global ptr
  ptr = 0
  model.feature_extractor.darknet53.apply(load_weights)
  model.feature_extractor.layers.apply(load_weights)
  
  err_msg = "There are weights that are never used in the convertion. This is likely due to the Pytorch model having a lower amount of classes than the orginial weights which is not allowed"
  assert weight_size == ptr, err_msg

  model.save_weights('converted.weights')

def check_download(weights_file):
  if not Path(weights_file).exists():
    return False

  def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
      for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    return hash_md5.hexdigest()

  weight_hash_gt = 'c84e5b99d0e52cd466ae710cadf6d84c'
  weight_hash = md5(weights_file)
  return weight_hash == weight_hash_gt

def download_weights(weights_file):
  print("Downloading weights...")
  url = 'https://pjreddie.com/media/files/yolov3.weights'
  wget.download(url, weights_file)

if __name__ == '__main__':
  weights_file = 'original_coco_416.weights'
  download_correct = check_download(weights_file)
  if not download_correct:
    download_weights(weights_file)
    download_correct = check_download(weights_file)
    err_msg = "Hashes differs. It seems like the download wasn't completed successfully or hashes has changed since project setup. Get the yolov3-416 weights from https://pjreddie.com/darknet/yolo/ and name them {}. ".format(weights_file)
    assert download_correct, err_msg
  

  model = Yolov3(80, NoCuda('default'))
  convert_weights(weights_file, model)
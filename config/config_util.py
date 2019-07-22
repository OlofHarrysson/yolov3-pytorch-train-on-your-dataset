import sys
from io import StringIO
import fire
from .base_config import *

def choose_config(config_str):
  if config_str == 'coco':
    config_obj = Coco(config_str)
  elif config_str == 'faces':
    config_obj = Faces(config_str)
  elif config_str == 'fruits':
    config_obj = Fruits(config_str)
  else:
    raise NotImplementedError('Add your config class')

  # Overwrite parameters via optional input flags
  return overwrite(config_obj)


def overwrite(config_obj):
  ''' Overwrites parameters with input flags. Function is needed for the convenience of specifying parameters via a combination of the config classes and input flags. '''

  class NullIO(StringIO):
    def write(self, txt):
      pass

  def parse_unknown_flags(**kwargs):
    return kwargs

  sys.stdout = NullIO()
  extra_arguments = fire.Fire(parse_unknown_flags)
  sys.stdout = sys.__stdout__

  for key, val in extra_arguments.items():
    if key not in vars(config_obj):
      err_str = "The input parameter {} isn't allowed. It's only possible to overwrite attributes that exist in the DefaultConfig class. Add your input parameter to the default class or catch it before this message".format(key)
      raise NotImplementedError(err_str)
    setattr(config_obj, key, val);

  return config_obj
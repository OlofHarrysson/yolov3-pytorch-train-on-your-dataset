import argparse
from models.yolov3 import Yolov3
from utils.controller import train
from utils.utils import load_classes, seed_program
from data.dataloader import Dataloader
from config.config_util import choose_config

def parse_args():
  p = argparse.ArgumentParser()

  configs = ['coco', 'faces', 'fruits']
  p.add_argument('--config', type=str, default='faces', choices=configs, help='What config file to choose')

  args, unknown = p.parse_known_args()
  return args.config

def main(config_str):
  config = choose_config(config_str)
  seed_program(config.seed)
  dataloader = Dataloader(config)

  # Load classes
  classes_json = 'datasets/{}/classes.json'.format(config.classdir)
  classes = load_classes(classes_json)
  n_classes = len(classes)

  # Create model
  model = Yolov3(n_classes, config)

  # Load & freeze weights
  if config.weights:
    load_last = not config.skip_last
    model.load_weights(config.weights, load_last=load_last)

  if config.weight_freeze:
    model.freeze_weights(config.weight_freeze)

  train(model, config, dataloader)

if __name__ == '__main__':
  config_str = parse_args()
  main(config_str)
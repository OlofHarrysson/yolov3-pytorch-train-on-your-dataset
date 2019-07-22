import argparse, math, shutil
from models.yolov3 import Yolov3
from utils.utils import load_classes, seed_program
from data.dataloader import Dataloader
from config.config_util import choose_config
from utils.validator import Validator
from eval.Evaluator import Evaluator
from pathlib import Path
import numpy as np
from PIL import Image
from eval.BoundingBoxes import BoundingBoxes
from collections import defaultdict

def parse_args():
  p = argparse.ArgumentParser()

  configs = ['coco', 'faces']
  p.add_argument('--config', type=str, default='faces', choices=configs, help='What config file to choose')

  args, unknown = p.parse_known_args()
  return args.config

def clear_output_dir():
  for outdir in Path('output').iterdir():
    try:
      outdir.unlink()
    except:
      shutil.rmtree(str(outdir))

def main(config_str):
  clear_output_dir()
  seed_program()

  config = choose_config(config_str)
  dataloader = Dataloader(config)

  # Create model
  classes_json = 'datasets/{}/classes.json'.format(config.datadir)
  classes = load_classes(classes_json)
  n_classes = len(classes)
  model = Yolov3(n_classes, config)

  # Load weights
  if config.weights:
    load_last = not config.skip_last
    model.load_weights(config.weights, load_last=load_last)

  model.to(model.device)
  predict(model, dataloader, config, classes)

def predict(model, dataloader, config, classes):
  validator = Validator(config, dataloader, model, logger=None)
  model.eval()

  for name, data in dataloader.get_vals().items():
    bboxes, losses, im_paths = validator.forward(model, data, name)
    metrics = validator._calc_bbox_metrics(name, bboxes, 0, losses)

    # Save all output, or failure cases
    # save_bboxes(name, bboxes, im_paths)
    save_failiure_cases(im_paths, bboxes, config, name)

    # Print metrics & plot precision recall curve
    print_metrics(name, metrics)
    # plot_precision_recall_curve(bboxes)

def print_metrics(name, metrics):
  metric_str = 'Dataset:{}\tAP/F1/Prec/Recall/TotalLoss: {:.8f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}'
  print(metric_str.format(name, metrics['ap'], metrics['f1'], metrics['precision'], metrics['recall'], -metrics['loss']))

def plot_precision_recall_curve(bboxes):
  evaler = Evaluator()
  evaler.PlotPrecisionRecallCurve(0, bboxes, showAP=True)

def save_failiure_cases(im_paths, bboxes, config, name):
  fail_cases = find_failure_cases(im_paths, bboxes, config)
  for fail_case in fail_cases:
    fail_name = '{}_{}'.format(name, fail_case)
    save_bboxes(fail_name, bboxes, fail_cases[fail_case])

def find_failure_cases(im_paths, all_bboxes, config):
  evaler = Evaluator()
  failiure_cases = defaultdict(lambda: [])

  for im_path in im_paths:
    im_name = Path(im_path).name
    bboxes = all_bboxes.getBoundingBoxesByImageName(im_name)
    bboxes = BoundingBoxes(bboxes)
    iou = config.iou_thresh
    all_results = evaler.GetPascalVOCMetrics(bboxes, IOUThreshold=iou)
    res = all_results[0] # TODO: Works for 1 class

    true_pos = res['total TP']
    false_pos = res['total FP']
    n_gt = res['total positives']
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / n_gt

    # Avoid the cases where no prediction is made -> prec=NaN
    if precision != 1 and not math.isnan(precision):
      failiure_cases['precision'].append(im_path)

    # Avoid the cases where no gt exist -> recall=NaN
    if recall != 1 and not math.isnan(recall):
      failiure_cases['recall'].append(im_path)

  return failiure_cases

def save_bboxes(data_name, bboxes, im_paths):
  for im_path in im_paths:
    im_name = Path(im_path).name
    im = np.array(Image.open(im_path))
    im = bboxes.drawAllBoundingBoxes(im, im_name)
    im = Image.fromarray(im)
    out_dir = 'output/{}/'.format(data_name)
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    im.save(out_dir + im_name)

if __name__ == '__main__':
  config_str = parse_args()
  main(config_str)
import json, random, torch
import numpy as np
from eval.BoundingBox import BoundingBox
from eval.utils import BBType
from pathlib import Path
from eval.utils import BBFormat
import imgaug as ia
import progressbar as pbar
from collections import defaultdict, namedtuple


def seed_program(seed=0):
  ''' Seed for reproducability '''
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  # torch.backends.cudnn.deterministic = True # You can add this

def load_classes(path):
  ''' Loads class labels at path '''
  with open(path) as infile:
    return json.load(infile)

def bbox_ious(bboxes1, bboxes2):
  x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
  x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
  xA = np.maximum(x11, np.transpose(x21))
  yA = np.maximum(y11, np.transpose(y21))
  xB = np.minimum(x12, np.transpose(x22))
  yB = np.minimum(y12, np.transpose(y22))
  interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
  boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
  boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
  iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
  return iou.squeeze(axis=0)

def bboxes_2_xyxy(bboxes):
  ''' bboxes are in the format of xywh and transforms to x1y1x2y2 '''
  x1 = bboxes[:, 0] - bboxes[:, 2] / 2 
  y1 = bboxes[:, 1] - bboxes[:, 3] / 2 
  x2 = bboxes[:, 0] + bboxes[:, 2] / 2 
  y2 = bboxes[:, 1] + bboxes[:, 3] / 2
  return np.stack((x1, y1, x2, y2), axis=1)

def handle_out_of_bounds(bboxes, image_size):
  ''' Make sure the coordinates are within the image '''
  return np.clip(bboxes, 0, image_size)

def non_max_suppression(predictions, conf_thres, nms_thres, image_size, data_helper):
  im_names = [Path(h.im_path).name for h in data_helper]
  best_bboxes = defaultdict(lambda: [])

  assert len(im_names) == len(predictions)
  # For every image in batch
  for im_name, image_pred in zip(im_names, predictions):

    # Filter out predictions with low obj_conf
    image_pred = image_pred[image_pred[..., 4] >= conf_thres]
    
    # If none are remaining => process next image
    if image_pred.size == 0:
      continue

    class_preds = image_pred[..., 5:]
    class_ints = np.argmax(class_preds, axis=1)
    found_classes = np.unique(class_ints)

    for class_int in found_classes: # For every class that was found
      # x, y, w, h, obj_conf, class_conf
      class_pred = image_pred[class_ints == class_int]

      # Sort by obj_conf. Start with the highest one
      sorted_preds = class_pred[class_pred[..., 4].argsort()][::-1]

      bboxes = bboxes_2_xyxy(sorted_preds[..., :4])
      bboxes = handle_out_of_bounds(bboxes, image_size)
      cls_conf = np.amax(sorted_preds[..., 5:], axis=1)
      obj_conf = sorted_preds[..., 4]

      # Perform non-maximum suppression
      max_detections = []
      confs = []
      conf_tup = namedtuple('Confidences', 'cls obj')
      while len(bboxes):
        max_detections.append(bboxes[0])
        confs.append(conf_tup(cls_conf[0], obj_conf[0]))
        last_box = max_detections[-1].reshape(1, -1)

        if len(bboxes) == 1:
          break

        ious = bbox_ious(last_box, bboxes[1:])

        # Filter out lower scored boxes that overlap
        bboxes = bboxes[1:][ious < nms_thres]
        cls_conf = cls_conf[1:][ious < nms_thres]
        obj_conf = obj_conf[1:][ious < nms_thres]

      for bbox, conf in zip(max_detections, confs):
        item = {'bbox': bbox, 'class_conf': conf.cls, 'class_pred': class_int, 'obj_conf': conf.obj}
        best_bboxes[im_name].append(item)

  return best_bboxes

def add_bboxes(bbox_preds, image_size, data_helper, bbox_builder):
  for h in data_helper:
    im_name = Path(h.im_path).name

    # Builds prediction boxes
    for box in bbox_preds[im_name]:
      img_shape = (image_size, image_size)
      in_box = box['bbox'].reshape(1, -1)
      bboxes = ia.BoundingBoxesOnImage.from_xyxy_array(in_box, img_shape)
      bbs = h.unpad.augment_bounding_boxes([bboxes])[0]
      x1, y1, x2, y2 = bbs.to_xyxy_array()[0]
      class_id = int(box['class_pred'])

      bbox_conf = box['obj_conf']
      bbox = BoundingBox(im_name, class_id, x1, y1, x2, y2, classConfidence=bbox_conf, bbType=BBType.Detected, format=BBFormat.XYX2Y2)
      bbox_builder.addBoundingBox(bbox)

    # Builds target boxes
    for target_box in h.target_bboxes:
      bbox_builder.addBoundingBox(target_box)

  return bbox_builder

class ProgressbarWrapper():
  def __init__(self, n_epochs, n_batches):
    self.text = pbar.FormatCustomText(
      'Epoch: %(epoch).d/%(n_epochs).d, Batch: %(batch)d/%(n_batches)d',
      dict(
        epoch=0,
        n_epochs=n_epochs,
        batch=0,
        n_batches=n_batches
      ),
    )

    self.bar = pbar.ProgressBar(widgets=[
        pbar.Percentage(), ' ',
        self.text, ' ',
        pbar.Bar(), ' ',
        pbar.Timer(), ' ',
        pbar.AdaptiveETA(), ' ',
    ], redirect_stdout=True)

    self.bar.start()

  def __call__(self, down_you_go):
    return self.bar(down_you_go)

  def update(self, epoch, batch):
    self.text.update_mapping(epoch=epoch, batch=batch)
    self.bar.update()
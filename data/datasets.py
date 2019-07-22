from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from models.dataclasses import DatasetHelper
from pathlib import Path
from eval.BoundingBox import BoundingBox
from eval.utils import BBFormat
from more_itertools import chunked

# TODO: make it so that tranforms can easily be added. normalize from absolute coords to relative (0-1), with or without padding
# TODO: Any benefit to encode image size and what not in label?
# TODO: Have option for no resize
# TODO: Need to do data augmentation like lighting, rotate before pad? Because otherwise the augmentation will also happen on the background fill values. Not good right?
# TODO: Create all non geometric transformations first, then rotate, in the end pad.


class ImageDataset(Dataset):
  def __init__(self, index_path, im_sizes, batch_size):
    self.im_sizes = im_sizes
    self.batch_size = batch_size
    with open(index_path, 'r') as file:
      self.image_files = file.read().splitlines()
    assert self.image_files,'{} dataset is empty'.format(index_path)

  def __len__(self):
    return len(self.image_files)

  def set_index_2_imsize(self, sampler):
    self.index_2_imsize = {}
    indexes = list(chunked(sampler, self.batch_size))

    for i, batch_indexes in enumerate(indexes):
      for im_index in batch_indexes:
        im_size_index = i % len(self.im_sizes)
        self.index_2_imsize[im_index] = self.im_sizes[im_size_index]

  def __getitem__(self, index):
    im_path = self.image_files[index]

    label_path = Path(im_path.replace("images/", "labels/"))
    label_stem = Path(label_path).stem
    label_path = '{}/{}.txt'.format(label_path.parent, label_stem)

    im = Image.open(im_path)
    if im.mode != 'RGB': # Handle black & white images
      im = im.convert(mode='RGB') 
    im = np.array(im)

    # Maps from index to image_size. Cycles thorugh the image sizes
    im_size = self.index_2_imsize[index]

    # Pad & Scale
    padding = ia.compute_paddings_for_aspect_ratio(im, 1.0)
    pad_fill = 128 # Halfway between black and white
    seq = iaa.Sequential([
      iaa.Pad(px=padding, pad_cval=pad_fill),
      iaa.Scale(im_size)
    ])

    # Perform transformations
    im_aug = seq.augment_image(im)

    # Invert transform
    h, w, _ = im.shape
    unpad = iaa.Sequential([
      iaa.Scale(max(h,w)),
      iaa.Crop(padding, keep_size=False)
    ])
      
    labels, target_bboxes = self._build_target(label_path, im.shape, seq, im_path, im_size)

    data_helper = DatasetHelper(im_path, target_bboxes, unpad)

    # Channels-first
    input_img = np.transpose(im_aug, (2, 0, 1))
    normalize = lambda arr: arr / np.float32(255)
    return normalize(input_img), labels, data_helper

  def _build_target(self, label_path, image_shape, seq, im_path, im_size):
    if Path(label_path).stat().st_size == 0:  # No gt
      return [], []
      
    labels_raw = np.loadtxt(label_path).reshape(-1, 5)

    coords = labels_raw[:, 1:]
    bboxes = ia.BoundingBoxesOnImage.from_xyxy_array(coords, image_shape)

    bbs_aug = seq.augment_bounding_boxes([bboxes])[0]

    # To relative & xywh format
    labels_aug = bbs_aug.to_xyxy_array()
    labels = np.zeros_like(labels_raw)
    labels[:, 1] = (labels_aug[:, 0] + labels_aug[:, 2]) / 2
    labels[:, 2] = (labels_aug[:, 1] + labels_aug[:, 3]) / 2
    labels[:, 3] = labels_aug[:, 2] - labels_aug[:, 0]
    labels[:, 4] = labels_aug[:, 3] - labels_aug[:, 1]
    labels /= im_size
    
    labels[:, 0] = labels_raw[:, 0] # Class int

    # Bounding boxes, # Invert transform
    image_name = Path(im_path).name
    target_bboxes = self._build_target_boxes(labels_raw, image_name)

    return labels, target_bboxes

  def _build_target_boxes(self, labels_raw, image_name):
    bboxes = []
    for target_box in labels_raw:
      class_id = int(target_box[0])
      x1, y1, x2, y2 = target_box[1:]
      bboxes.append(BoundingBox(image_name, class_id, x1, y1, x2, y2, format=BBFormat.XYX2Y2))

    return bboxes


def augment(im, labels):
  # Do augmentations. Return a seq
  # Add bboxes on image
  # Add non geometric transformations
  # Add geometric transformations
  # Add padding + resize
  # Create bboxes for helper
  pass

  # seq_det = seq.to_deterministic()

import torch, math
from eval.Evaluator import Evaluator
from eval.BoundingBoxes import BoundingBoxes
from utils.utils import add_bboxes, non_max_suppression
import numpy as np

class Validator():
  def __init__(self, config, dataloader, model, logger):
    self.validation_sets = dataloader.get_vals()
    self.model, self.config, self.logger = model, config, logger

    criterias = config.weight_save_criteria
    self.best_metrics = {key:float('-inf') for key in criterias}

    self.conf_thresh = self.config.confidence_thresh
    self.nms_thresh = self.config.non_maximal_supression_thresh
    self.max_val_batches = self.config.max_validation_batches

  def print_metrics(self, metrics, name, epoch):
    names = self.validation_sets.keys()
    pad = len(max(list(names), key=len))

    metric_str = 'Epoch: {}\t Data: {:<{width}s}\t AP/F1/Prec/Recall/TotalLoss: {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}'
    print(metric_str.format(epoch, name, metrics['ap'], metrics['f1'], metrics['precision'], metrics['recall'], -metrics['loss'], width=pad))

  def validate(self, step, epoch):
    print("~~~~~~~~ Started Validation ~~~~~~~~")
    model = self.model
    model.eval()
    for name, data in self.validation_sets.items():
      bboxes, losses, im_paths = self.forward(model, data, name)
      metrics = self._calc_bbox_metrics(name, bboxes, step, losses)
      self.logger.save_losses(losses, name, step)
      self.logger.save_metrics(metrics, name)
      self.print_metrics(metrics, name, epoch)
      self.save_on_improvement(name, metrics)

    model.train()
    print("~~~~~~~~ Finished Validation ~~~~~~~~")

  def forward(self, model, val_set, data_name):
    losses, im_paths = [], []
    bboxes = BoundingBoxes()

    n_batches = min(self.max_val_batches, len(val_set))
    for batch_i, data in enumerate(val_set, 1):
      if batch_i > n_batches:
        break

      inputs, labels, data_helper = data
      [im_paths.append(h.im_path) for h in data_helper]
      image_size = inputs.size(2)

      with torch.no_grad():
        outputs = model(inputs)
        preds = model.predict(outputs, image_size)
        losses.append(model.calc_loss(outputs, labels, image_size))

      bbox_preds = non_max_suppression(preds.numpy(), self.conf_thresh, self.nms_thresh, image_size, data_helper)
      bboxes = add_bboxes(bbox_preds, image_size, data_helper, bboxes)

    return bboxes, losses, im_paths

  def _calc_bbox_metrics(self, data_name, bboxes, step, losses):
    ''' Calculates metrics for the bounding boxes. Currently averages the calculations over classes '''
    evaler = Evaluator()

    # Calculates the negative loss (want to maximize it)
    negative_loss = 0
    for loss in losses:
      negative_loss -= sum(loss.values()).item()
    negative_loss /= len(losses)

    metrics_per_class = []
    iou = self.config.iou_thresh
    all_results = evaler.GetPascalVOCMetrics(bboxes, IOUThreshold=iou)

    # Calculates metrics
    for res in all_results:
      true_pos = res['total TP']
      false_pos = res['total FP']
      n_gt = res['total positives']

      safe_divide = lambda n, d, fallback: n / d if d else fallback
      precision = safe_divide(true_pos, true_pos + false_pos, 0)
      recall = safe_divide(true_pos, n_gt, 1)
      f1 = safe_divide(2 * precision * recall, precision + recall, 0)
      ap = res['AP'] if not math.isnan(res['AP']) else 0

      metric = [precision, recall, f1, ap]
      metrics_per_class.append(metric)

    # Calculates mean metrics
    metrics_per_class = np.array(metrics_per_class, dtype=np.float32)
    mean_metrics = mm = metrics_per_class.mean(axis=0)
    return {'precision':mm[0], 'recall':mm[1], 'f1':mm[2], 'ap':mm[3], 'optim_step':step, 'loss': negative_loss}

  def save_on_improvement(self, name, metrics):
    ''' Saves model if it's better than the best recorded step '''
    save_dir = self.logger.save_path
    if name == self.config.weight_save_data:
      for metric in self.best_metrics.keys():
        score = metrics[metric]
        if score > self.best_metrics[metric]:
          self.best_metrics[metric] = score
          path = '{}/best_{}.weights'.format(save_dir, metric)
          self.model.save_weights(path)

    # TODO: Save optimizer? How can one do that and easily load again
    # optimizer.save_state_dict(save_dir + '/optimizer.pt')
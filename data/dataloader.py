from .datasets import ImageDataset
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import random
from collections import OrderedDict

class Dataloader():
  def __init__(self, config):
    sample = 'sample_' if config.sample else ''

    index_base = 'data/indexes/image_indexes/{}'.format(sample)
    train_path = '{}{}_train.txt'.format(index_base, config.datadir)
    im_sizes = config.train_image_sizes
    batch_size = config.batch_size
    self.train_data = ImageDataset(train_path, im_sizes, batch_size)

    train_sampler = RandomSampler(self.train_data)
    self.train_sampler = train_sampler

    n_workers = config.data_read_threads
    self.train_loader = DataLoader(self.train_data, batch_size=config.batch_size, num_workers=n_workers, collate_fn=my_collate, sampler=train_sampler)

    eval_im_sizes = config.eval_image_sizes
    self.val_loaders = {}
    self.dataset_and_sampler = []
    for val_set in config.validations:
      val_path = '{}{}.txt'.format(index_base, val_set)
      val_data = ImageDataset(val_path, eval_im_sizes, batch_size)
      val_sampler = RandomSampler(val_data)
      loader = DataLoader(val_data, batch_size=config.batch_size, num_workers=n_workers, collate_fn=my_collate, sampler=val_sampler)
      self.val_loaders[val_set] = loader
      self.dataset_and_sampler.append((val_data, val_sampler))

  def get_train(self):
    self.seed_train_sampler()
    return self.train_loader

  def get_vals(self):
    self.seed_val_samplers()
    return OrderedDict(sorted(self.val_loaders.items()))

  def seed_train_sampler(self):
    self.train_sampler.set_seed()
    self.train_data.set_index_2_imsize(iter(self.train_sampler))

  def seed_val_samplers(self):
    for data, sampler in self.dataset_and_sampler:
      sampler.set_seed()
      data.set_index_2_imsize(iter(sampler))


class RandomSampler(Sampler):
  def __init__(self, data_source):
    self.data_source = data_source

  def set_seed(self):
    self.seed = random.randint(0, 2**32 - 1)

  def __iter__(self):
    n = len(self.data_source)
    indexes = list(range(n))
    random.Random(self.seed).shuffle(indexes)
    return iter(indexes)

  def __len__(self):
    return len(self.data_source)


def my_collate(batch):
  to_torch = lambda x: torch.as_tensor(x)
  images = torch.stack([to_torch(item[0]) for item in batch], 0)
  labels = [item[1] for item in batch]
  data_helper = [item[2] for item in batch]
  return (images, labels, data_helper)
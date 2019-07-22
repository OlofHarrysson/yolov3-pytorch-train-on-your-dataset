from pathlib import Path
import subprocess as subp

def du(path):
  """disk usage in human readable format (e.g. '2,1GB')"""
  return subp.check_output(['du','-sh', path]).split()[0].decode('utf-8')

def bbox_stats(label_dir):
  n_bboxes = 0
  for path in label_dir.iterdir():
    with open(str(path)) as f:
      bboxes = f.read().splitlines()
      n_bboxes += len(bboxes)

  return n_bboxes

def print_info(data_dir):
  size = du(str(data_dir))
  im_dir = data_dir / 'images'
  n_images = len(list(im_dir.iterdir()))
  n_bboxes = bbox_stats(data_dir / 'labels')

  dataset = data_dir.parent.name + ' ' + data_dir.name 

  print_string = 'Number of Images: {}\nNumber of Boundnig Boxes: {}\nSize: {}'

  print('~~~~~~~~~~~~~ {} ~~~~~~~~~~~~~'.format(dataset))
  print(print_string.format(n_images, n_bboxes, size))

def main():
  dataset_dir = Path('../datasets')

  # For every folder in the datasets dir
  for dataset in dataset_dir.iterdir():
    if dataset.name.startswith('.'): # Skip .gitignore
      continue

    # For train/val/test in dataset
    for datafolder in dataset.iterdir():
      if datafolder.is_dir():
        print_info(datafolder)


if __name__ == '__main__':
  main()
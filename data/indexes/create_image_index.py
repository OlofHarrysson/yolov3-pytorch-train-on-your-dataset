from pathlib import Path
import os, argparse, random
from PIL import Image

def parse_args():
  p = argparse.ArgumentParser()

  p.add_argument('-d', '--datadir', type=str, required=True, help='The data to create an index for. E.g. coco or another directory in the dataset folder')

  config = ['train', 'val', 'test']
  p.add_argument('-m', '--mode', type=str, required=True, choices=config, help='The data type')

  p.add_argument('-l', '--limit', type=int, default=50, help='Max number of images in sample')

  args = p.parse_args()
  return args

def assert_data(im_paths, label_paths):
  im_stems = [p.stem for p in im_paths]
  label_stems = [p.stem for p in label_paths]
  
  # Tests
  check_duplicates = lambda names: len(set(names)) == len(names)
  same_length = lambda x, y: len(x) == len(y)
  verify_ims = lambda paths: [Image.open(p).close() for p in paths]
  verify_label_suffix = lambda lbs: [l.suffix == '.txt' for l in lbs]
  im_label_pairs = lambda ims, lbs: set(ims) == set(lbs)

  # Asserts
  verify_ims(im_paths)
  assert all(verify_label_suffix(label_paths)), 'Labels need to end in .txt'
  assert check_duplicates(im_stems), 'There are images with the same name but different suffix. Not allowed'
  assert same_length(im_stems, label_stems), 'There are different amounts of images and labels'
  assert im_label_pairs(im_stems, label_stems), "The image/label pairs doesn't have the same name"

def create_index(args):
  this_dir = os.getcwd()
  dataset_dir = Path(this_dir).parent.parent / 'datasets'
  data_dir = dataset_dir / Path(args.datadir) / Path(args.mode)

  is_hidden_file = lambda path: path.name[0] == '.'

  im_dir = data_dir / 'images'
  im_paths = list(Path(im_dir).iterdir())
  im_paths = [p for p in im_paths if not is_hidden_file(p)]

  label_dir = data_dir / 'labels'
  label_paths = list(Path(label_dir).iterdir())
  label_paths = [p for p in label_paths if not is_hidden_file(p)]

  assert_data(im_paths, label_paths)

  out_path = 'image_indexes/{}_{}.txt'.format(args.datadir, args.mode)
  with open(out_path, 'w') as f:
    f.writelines("\n".join(map(str, im_paths)))

def create_sample(args):
  file_name = '{}_{}.txt'.format(args.datadir, args.mode)
  in_path = 'image_indexes/{}'.format(file_name)
  out_path = 'image_indexes/sample_{}'.format(file_name)

  with open(in_path) as f:
    lines = f.read().splitlines()

  random.shuffle(lines)
  lines = lines[:args.limit]

  with open(out_path, 'w') as f:
    f.writelines("\n".join(lines))

if __name__ == '__main__':
  args = parse_args()
  print("Creating Index...")
  create_index(args)

  print("Creating sample...")
  create_sample(args)

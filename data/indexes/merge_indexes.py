from pathlib import Path
import argparse, random

def parse_args():
  p = argparse.ArgumentParser()

  p.add_argument('-i1', '--index1', type=str, required=True, help='One index to merge')

  p.add_argument('-i2', '--index2', type=str, required=True, help='Second index to merge')

  p.add_argument('-o', '--output', type=str, required=True, help='Output name')

  p.add_argument('-l', '--limit_1', type=int, help='Max number of images to use from index1')

  p.add_argument('-s', '--sample', action="store_true", help='Also create a sample index')

  p.add_argument('-m', '--limit_sample', type=int, default=50, help='Max number of images in sample')

  args = p.parse_args()
  return args

def read_ind(path):
  with open(path) as f:
    return f.read().splitlines()

def merge_index(args):

  ind1_path = 'image_indexes/{}.txt'.format(args.index1)
  ind2_path = 'image_indexes/{}.txt'.format(args.index2)

  assert Path(ind1_path).is_file()
  assert Path(ind2_path).is_file()

  ind1 = read_ind(ind1_path)
  ind2 = read_ind(ind2_path)

  # Filters out a number of images from index one to create ratios
  if args.limit_1:
    ind1 = ind1[:args.limit_1]

  merged_index = ind1 + ind2
  copy = merged_index

  out_path = 'image_indexes/{}.txt'.format(args.output)
  with open(out_path, 'w') as f:
    f.writelines("\n".join(map(str, merged_index)))

  return copy

def create_sample(index, args):
  out_path = 'image_indexes/sample_{}'.format(args.output)

  random.shuffle(index)
  index = index[:args.limit_sample]

  with open(out_path, 'w') as f:
    f.writelines("\n".join(index))


if __name__ == '__main__':
  args = parse_args()
  print("Creating Index...")
  index = merge_index(args)

  if args.sample:
    print("Creating sample...")
    create_sample(index, args)
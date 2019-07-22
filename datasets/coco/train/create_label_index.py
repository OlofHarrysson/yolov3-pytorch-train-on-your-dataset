from pathlib import Path
import os

paths = []
for im_path in Path("images/").glob('**/*.jpg'):
  p = im_path.name
  p = os.getcwd() + '/labels/' + p.replace('.jpg', '.txt')
  paths.append(p)

with open('label_index.txt', 'w') as f:
  f.writelines("\n".join(paths))
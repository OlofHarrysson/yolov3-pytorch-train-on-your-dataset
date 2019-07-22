from pathlib import Path
import os

paths = []
for im_path in Path("images/").glob('**/*.jpg'):
  p = im_path.name
  p = os.getcwd() + '/labels/' + p.replace('.jpg', '.txt')

  try:
    Path(p).touch(exist_ok=False)
  except:
    # Label has something in it and should not be created
    pass
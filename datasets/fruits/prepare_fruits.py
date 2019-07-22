from pathlib import Path
import xml.etree.ElementTree as ET
import shutil, json

def labels_from_file(label_file):
  mapping = dict(apple=0, banana=1, orange=2)

  root = ET.parse(str(label_file)).getroot()
  objects = root.findall('object')

  coords = []
  for obj in objects:
    class_ = obj.find('name').text
    class_ = str(mapping[class_])
    bbox = obj.find('bndbox')
    xmin = bbox.find('xmin').text
    ymin = bbox.find('ymin').text
    xmax = bbox.find('xmax').text
    ymax = bbox.find('ymax').text
    label = [class_, xmin, ymin, xmax, ymax]
    coords.append(' '.join(label))

  return coords

def format_directory(dir_path):
  im_dir = dir_path / Path('images') 
  label_dir = dir_path / Path('labels') 
  im_dir.mkdir(exist_ok=True)
  label_dir.mkdir(exist_ok=True)
  
  for label_file in dir_path.iterdir():
    if label_file.suffix != '.xml':
      continue
    labels = labels_from_file(label_file)
    im_file_name = str(label_file).replace('.xml', '.jpg')
    im_out_path = '{}/images/{}'.format(dir_path, Path(im_file_name).name)
    shutil.move(im_file_name, im_out_path)

    label_out_path = im_out_path.replace('/images/', '/labels/')
    label_out_path = label_out_path.replace('.jpg', '.txt')
    with open(label_out_path, 'w') as f:
      f.writelines("\n".join(labels))

    label_file.unlink()

def check_data_exists(data_path):
  if not data_path.is_dir():
    err_msg = "Seems like there is no data to convert. Either get the data from https://www.kaggle.com/mbkinaci/fruit-images-for-object-detection and extract it and any subfolders here + name it fruit-images-for-object-detection. Alternativly download the prepared data from here https://drive.google.com/open?id=1XCndSDkB98WSZMjMQzdfk8ZgpP7mjF-g"
    assert False, err_msg


def make_class_index():
  mapping = dict(apple=0, banana=1, orange=2)
  with open('classes.json', 'w') as outfile:
    json.dump(mapping, outfile, indent=2)

def main():
  make_class_index()
  annotation_dir = Path('fruit-images-for-object-detection')
  check_data_exists(annotation_dir)
  
  train_dir = annotation_dir / Path('train')
  test_dir = annotation_dir / Path('test')
  shutil.move(str(train_dir), '.')
  shutil.move(str(test_dir), '.')
  Path('test').rename('val')

  # Format to out format
  format_directory(Path('train'))
  format_directory(Path('val'))

  # Clean up
  for subdir in annotation_dir.iterdir():
    subdir.unlink()
  annotation_dir.rmdir()

if __name__ == '__main__':
  main()
from pathlib import Path
import json, wget, sys
from progressbar import progressbar
from checksumdir import dirhash
from bs4 import BeautifulSoup
import requests

# ~~~~~~~~~~~~~~~~~ Download ~~~~~~~~~~~~~~~~~
def download_index(url_file):
  def make_soup(file_id):
      response = requests.get('http://s000.tinyupload.com/?file_id=' + file_id)
      soup = BeautifulSoup(response.text, 'html.parser')
      return soup


  def get_filelink(soup):
      for x in soup.find_all('a'):
          if x['href'].startswith('download.php'):
              full_link = 'http://s000.tinyupload.com/' + x['href']
              return full_link

  def download(link, outfile):
    wget.download(link, outfile, bar=None)

  try:
    file_id = '99685816638231149864'
    soup = make_soup(file_id)
    link = get_filelink(soup)
    download(link, url_file)
  except:
    print("Couldn't download the index file. Try downloading it manually and put it in this folder with the name face_detection.json from http://s000.tinyupload.com/index.php?file_id=99685816638231149864, https://www.kaggle.com/dataturks/face-detection-in-images or https://dataturks.com/projects/devika.mishra/face_detection/export")
    sys.exit(1)


def download_data():
  print("Downloading data...")
  url_file = 'face_detection.json'
  if not Path(url_file).exists():
    print("Downloading index...")
    download_index(url_file)

  data = []
  with open('face_detection.json') as infile:
    for line in infile:
      data.append(json.loads(line))

  for i, data_info in enumerate(progressbar(data)):
    im_url = data_info['content']
    im_info = data_info['annotation'][0]
    im_height = im_info['imageHeight']
    im_width = im_info['imageWidth']

    labels = []
    for annotation in data_info['annotation']:
      coords = annotation['points']
      x1 = coords[0]['x'] * im_width
      y1 = coords[0]['y'] * im_height
      x2 = coords[1]['x'] * im_width
      y2 = coords[1]['y'] * im_height
      label = '0 {} {} {} {}'.format(x1, y1, x2, y2)
      labels.append(label)

    label_path = 'labels/{}.txt'.format(i)
    with open(label_path, 'w') as f:
      f.writelines("\n".join(labels))

    im_path = 'images/{}.jpg'.format(i)
    wget.download(im_url, im_path, bar=None)



# ~~~~~~~~~~~~~~~~~~ Utils ~~~~~~~~~~~~~~~~~
def check_download():
  Path('images').mkdir(exist_ok=True)
  Path('labels').mkdir(exist_ok=True)
  im_hash_gt = '7278d82a532ac8987b210df4c4f54ab7'
  label_hash_gt = '58aec913ac05db6db7f4971310585917'
  im_hash = dirhash('images', 'md5')
  label_hash = dirhash('labels', 'md5')

  return im_hash == im_hash_gt and label_hash == label_hash_gt

def clear_output_dirs():
  for outdir in Path('images').iterdir():
    outdir.unlink()
  for outdir in Path('labels').iterdir():
    outdir.unlink()

def create_classes_file():
  classes = dict(face = 0)
  with open('classes.json', 'w') as outfile:
    json.dump(classes, outfile, indent=2)

# ~~~~~~~~~~~~~~~~~~ CleanUp ~~~~~~~~~~~~~~~~~
def split_data():
  print("Moving data...")
  Path('train/images').mkdir(exist_ok=True, parents=True)
  Path('train/labels').mkdir(exist_ok=True, parents=True)

  Path('val/images').mkdir(exist_ok=True, parents=True)
  Path('val/labels').mkdir(exist_ok=True, parents=True)

  im_paths = list(Path('images').iterdir())
  val_paths = im_paths[:15]
  train_paths = im_paths[15:]

  for train_path in train_paths:
    label_path = str(train_path).replace('images', 'labels')
    label_path = label_path.replace('.jpg', '.txt')
    name = train_path.stem

    Path(train_path).replace('train/images/{}.jpg'.format(name))
    Path(label_path).replace('train/labels/{}.txt'.format(name))

  for val_path in val_paths:
    label_path = str(val_path).replace('images', 'labels')
    label_path = label_path.replace('.jpg', '.txt')
    name = val_path.stem
    Path(val_path).replace('val/images/{}.jpg'.format(name))
    Path(label_path).replace('val/labels/{}.txt'.format(name))

def clean_up():
  print("Deleting unused directorys + files...")
  Path('images').rmdir()
  Path('labels').rmdir()
  Path('face_detection.json').unlink()

if __name__ == '__main__':
  download_correct = check_download()
  if not download_correct:
    clear_output_dirs()
    download_data()
    download_correct = check_download()
    err_msg = "Hashes differs. It seems like the download wasn't completed successfully or hashes has changed since project setup"
    assert download_correct, err_msg

  create_classes_file()
  split_data()
  clean_up()
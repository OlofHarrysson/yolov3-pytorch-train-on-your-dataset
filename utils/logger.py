from datetime import datetime
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

class Logger():
  def __init__(self, config):
    self.save_path = "saved/runs/{}".format(config.start_time)
    Path(self.save_path).mkdir(parents=True)
    for val_data in config.validations:
      save_dir = '{}/{}'.format(self.save_path, val_data)
      Path(save_dir).mkdir(parents=True)

    self.save_flags(config)

  def save_flags(self, config):
    ''' Saves config parameters '''
    params = config.get_parameters()
    with open('{}/parameters.json'.format(self.save_path), 'w') as f:
      json.dump(params, f,  indent=2)


  def save_metrics(self, data, name):
    df = pd.DataFrame(data, index=[0])

    csv_path = '{}/{}/metrics.csv'.format(self.save_path, name)
    with open(csv_path, 'a') as f:
      df.to_csv(f, mode='a', index=False, header=f.tell()==0)


  def save_losses(self, losses, name, step):
    dict_losses = defaultdict(list)
    for loss in losses:
      for loss_name, v in loss.items():
        dict_losses[loss_name].append(v.item())
    
    df = pd.DataFrame(dict_losses)
    df['optim_step'] = step

    csv_path = '{}/{}/loss.csv'.format(self.save_path, name)
    with open(csv_path, 'a') as f:
      df.to_csv(f, mode='a', index=False, header=f.tell()==0)


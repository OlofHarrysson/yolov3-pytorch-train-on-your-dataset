from pathlib import Path
import visdom, json, argparse
from layout import build_everything_view, save_layout, build_view
from plots import PlotBuilder
from itertools import chain
from collections import defaultdict

def read_params(run_dir):
  param_path = '{}/parameters.json'.format(run_dir)
  with open(param_path) as infile:
    return json.load(infile)

def clear_env(vis, env_name):
  ''' Clear between runs to not populate visdom with windows from previous runs '''
  vis.close(env=env_name) # Kills wind
  vis.delete_env(env_name) # Kills envs

def parse_args():
  p = argparse.ArgumentParser()

  p.add_argument('-c','--compare', nargs='+', help='List of runs to compare.', required=True)

  p.add_argument('--comment', type=str, default='I have nothing to say.', help='Optional comment')

  args = p.parse_args()
  return args

def build_compare_layout(datasets, run_dirs, window_titles):
  views = []
  add_view = lambda v: views.append(v)

  add_view(build_everything_view(window_titles.copy()))

  to_not_show = list(window_titles.values())
  to_not_show = set(chain.from_iterable(to_not_show))

  dataset_to_toshow = defaultdict(lambda: [])
  for dataset in datasets:
    to_show = [t for t in to_not_show if t.endswith(dataset)]
    for title in to_show:
      dataset_to_toshow[dataset].append(title)

  for data_name, titles in dataset_to_toshow.items():
    not_show = list(to_not_show - set(titles))
    add_view({data_name: build_view(titles, not_show)})

  save_layout(views, run_dirs)

if __name__ == '__main__':
  vis = visdom.Visdom()
  env_name = 'Compare Metrics'
  clear_env(vis, env_name)
  config = parse_args()
  run_dir = '../saved/runs_to_show/'

  run_dirs = []

  # If we specify a group of runs
  if len(config.compare) == 1 and config.compare[0].endswith('/'):
    all_dirs = list(Path(run_dir).iterdir())
    start = config.compare[0][:-1]
    run_dirs = [d for d in all_dirs if d.name.startswith(start)]

  # If we specify a list of runs
  else:
    for to_compare in config.compare:
      run_dirs.append(run_dir / Path(to_compare))

  plot_builder = PlotBuilder(vis)

  for run_dir in run_dirs:
    run_name = run_dir.name
    run_name = env_name
    save_params = read_params(run_dir)
    info = {'optional_comment': config.comment}
    plot_builder.parameters(info, run_name)

    datasets = save_params['validations']

    for data_name in datasets:
      data_dir = '{}/{}/'.format(run_dir, data_name)
      data_name = str(run_dir.name) + '_' + data_name
      plot_builder.metrics(data_dir, run_name, data_name, title_base='Bbox')

  titles = plot_builder.window_titles
  build_compare_layout(datasets, [Path(env_name)], titles)
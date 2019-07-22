from pathlib import Path
import visdom, json, argparse
from layout import build_layout
from plots import PlotBuilder

def clear_envs(vis):
  [vis.close(env=env) for env in vis.get_env_list()] # Kills wind
  [vis.delete_env(env) for env in vis.get_env_list()] # Kills envs

def read_params(run_dir):
  param_path = '{}/parameters.json'.format(run_dir)
  with open(param_path) as infile:
    return json.load(infile)

def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('-s', '--show', action="store_true", help='Plot from runs_to_show folder')
  args = p.parse_args()
  return args

if __name__ == '__main__':
  vis = visdom.Visdom()
  # clear_envs(vis)
  # print(vis.get_env_list())
  config = parse_args()
  what_dir = 'runs_to_show/' if config.show else 'runs/'
  run_dir = '../saved/{}'.format(what_dir)
  run_dirs = list(Path(run_dir).iterdir())
  hidden_file = lambda f: str(f.name)[0] == '.' # Remove hidden file on Mac
  run_dirs = list(filter(lambda d: not hidden_file(d), run_dirs))
  plot_builder = PlotBuilder(vis)

  for run_dir in run_dirs:
    run_name = run_dir.name
    save_params = read_params(run_dir)
    plot_builder.parameters(save_params, run_name)

    datasets = save_params['validations']
    try:
      plot_builder.train_vs_val(datasets, run_dir)
    except Exception as e:
      print(str(e) + ' -> Skipping train vs val view')

    for data_name in datasets:
      data_dir = '{}/{}/'.format(run_dir, data_name)
      plot_builder.dataset_plot(data_dir, run_name, data_name)
      

  titles = plot_builder.window_titles
  build_layout(datasets, run_dirs, titles)
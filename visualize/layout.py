import json
from itertools import chain
from pathlib import Path

def build_layout(datasets, run_dirs, window_titles):
  views = []
  add_view = lambda v: views.append(v)

  add_view(build_everything_view(window_titles.copy()))
  add_view(build_train_vs_val_view(window_titles.copy()))
  add_view(build_loss_view(window_titles.copy()))
  add_view(build_metric_view(window_titles.copy()))

  for dataset in datasets:
    add_view(build_data_view(dataset, window_titles.copy()))

  save_layout(views, run_dirs)

def save_layout(views, run_dirs):
  layout = {}
  tot_view = {}
  for view in views:
    tot_view.update(view)

  for run_dir in run_dirs:
    run_name = run_dir.name
    layout[run_name] = tot_view

  # Insert boilerplate
  layout['main'] = {}
  for key, val in layout.items():
    val['current'] = {}

  # print(json.dumps(layout, indent=2))
  layout_path = Path.home() / '.visdom/view/layouts.json'
  with open(layout_path, 'w') as outfile:
    json.dump(layout, outfile, indent=2)

def build_everything_view(windows):
  to_show = list(windows.values())
  to_show = sorted(list(set(chain.from_iterable(to_show))))

  first_element = 'parameters'
  to_show.remove(first_element)
  to_show.insert(0, first_element)

  view = {}
  height = 30
  width = 35
  for i, window in enumerate(to_show):
    view[window] = [i, height, width]

  return {'Everything': view}

def build_view(to_show, to_not_show):
  view = {}
  height = 30
  width = 35
  for i, window in enumerate(to_show + to_not_show):
    if i < len(to_show):
      view[window] = [i, height, width]
    else:
      view[window] = [i, 1, 1]

  return view

def to_show_or_not_to_show(windows, to_show):
  to_not_show = list(windows.values())
  to_not_show = set(chain.from_iterable(to_not_show))
  return list(to_not_show - set(to_show))

def build_data_view(data_name, windows):
  to_show = windows.pop(data_name)
  to_not_show = to_show_or_not_to_show(windows, to_show)

  return {data_name: build_view(to_show, to_not_show)}

def build_loss_view(windows):
  to_show = windows.pop('loss')
  to_not_show = to_show_or_not_to_show(windows, to_show)

  return dict(Loss=build_view(to_show, to_not_show))

def build_metric_view(windows):
  to_show = windows.pop('metrics')
  to_not_show = to_show_or_not_to_show(windows, to_show)

  return dict(Metrics=build_view(to_show, to_not_show))

def build_train_vs_val_view(windows):
  to_show = windows.pop('train_vs_val')
  to_not_show = to_show_or_not_to_show(windows, to_show)

  build_view(to_show, to_not_show)
  return {'Train vs Val': build_view(to_show, to_not_show)}

import pandas as pd
import plotly.graph_objs as go
from collections import defaultdict, OrderedDict
import functools, sys
from plot_utils import build_base_layout, build_loss_trace_with_std, build_max_metrics, plotly_color_map

class PlotBuilder():
  def __init__(self, visdom):
    self.vis = visdom
    self.window_titles = defaultdict(list)

  def dataset_plot(self, data_dir, run_name, data_name):
    self.metrics(data_dir, run_name, data_name)
    self.area(data_dir, run_name, data_name)
    self.loss_line(data_dir, run_name, data_name)

  def _add_views(func):
    @functools.wraps(func)
    def wrap(self, *args, **kwargs):
      views, win_id = func(self, *args, **kwargs)
      for view in views:
        self.window_titles[view].append(win_id)
    return wrap

  @_add_views
  def parameters(self, params, run_name):
    func_name = sys._getframe().f_code.co_name
    win_id = '{}'.format(func_name)
    params = OrderedDict(sorted(params.items()))

    # Makes a list of parameters in html
    vals = map(str, params.values())
    pairs_as_str =  [k+': '+v for k,v in zip(params, vals)]
    html = '</li><li>'.join(pairs_as_str)
    html = '<ul><li>{}</li></ul>'.format(html)

    opts = dict(title='Parameters')
    self.vis.text(html, win=win_id, env=run_name, opts=opts)
    views = ('none',) # Only adds in everything view
    return views, win_id

  @_add_views
  def train_vs_val(self, datasets, run_dir):
    train_set = [d for d in datasets if 'train' in d][0]
    val_set = train_set.replace('train', 'val')
    assert val_set in datasets, "Can't find validation dataset with same name as train"

    train_loss_csv = '{}/{}/loss.csv'.format(run_dir, train_set)
    train_df = pd.read_csv(train_loss_csv, index_col='optim_step')
    train_df = train_df.sum(axis=1).groupby('optim_step', axis=0)
    train_std = train_df.std().values
    train_mean = train_df.mean()
    train_invis = train_mean.values - train_std / 2
    xs = train_mean.index.values

    val_loss_csv = '{}/{}/loss.csv'.format(run_dir, val_set)
    val_df = pd.read_csv(val_loss_csv, index_col='optim_step')
    val_df = val_df.sum(axis=1).groupby('optim_step', axis=0)
    val_std = val_df.std().values
    val_mean = val_df.mean()
    val_invis = val_mean.values - val_std / 2

    train_data = build_loss_trace_with_std(xs, train_mean, train_std, train_invis, stack='one', name_prefix='Train ', color='31,119,180')
    val_data = build_loss_trace_with_std(xs, val_mean, val_std, val_invis, stack='two', name_prefix='Val ', color='255,127,14')

    func_name = sys._getframe().f_code.co_name
    win_id = '{}'.format(func_name)

    title = 'Train vs Val'
    layout = build_base_layout(title)
    overwrite = go.Layout(
      legend=dict(
        y=1.1,
        orientation="h",
        bgcolor='rgba(0,0,0,0)')
    )
    layout.update(overwrite)

    data = train_data + val_data
    fig = dict(data=data, layout=layout)
    run_name = run_dir.name
    self.vis.plotlyplot(fig, win=win_id, env=run_name)

    views = ('train_vs_val',)
    return views, win_id

  @_add_views
  def metrics(self, data_dir, run_name, data_name, title_base=None):
    csv_path = data_dir + 'metrics.csv'
    df = pd.read_csv(csv_path, index_col='optim_step')
    # Drop the loss column since we don't want to show it here
    df = df.drop(columns='loss', errors='ignore')

    max_metrics = df.max()
    max_xs = df.idxmax()
    legends = list(df.columns.values)
    xs = df.index.values
    data = []

    # Line graph
    for ys, legend in zip(df.values.T, legends):
      data.append(dict(
        x=xs,
        y=ys,
        hoverinfo='x+y',
        type='lines', # Visdom needs this
        mode='lines',
        name=legend,
      ))

    # Max points
    data.append(dict(
      x=max_xs,
      y=max_metrics,
      hoverinfo='x',
      mode='markers',
      showlegend=False,
      marker = dict(
        size=8,
        color = plotly_color_map()[:len(max_xs)],
        opacity=0.8
      )
    ))

    if title_base:
      title = '{} {}'.format(title_base, data_name)
    else:
      title = 'Bounding Box Metrics {}'.format(data_name)
    
    layout = build_base_layout(title)
    overwrite = go.Layout(
      legend=dict(
        y=1.1,
        orientation="h",
        bgcolor='rgba(0,0,0,0)'),
      annotations=build_max_metrics(max_metrics, legends)
    )
    layout.update(overwrite)

    func_name = sys._getframe().f_code.co_name
    win_id = '{} {}'.format(func_name, data_name)
    fig = dict(data=data, layout=layout)
    self.vis.plotlyplot(fig, win=win_id, env=run_name)

    views = ('metrics', data_name)
    return views, win_id

  @_add_views
  def area(self, data_dir, run_name, data_name):
    csv_path = data_dir + 'loss.csv'
    df = pd.read_csv(csv_path)

    losses = df.groupby('optim_step', axis=0).mean()
    losses = losses.div(losses.sum(axis=1), axis=0)

    xs = losses.index.values
    data = []
    for loss_name, loss in losses.iteritems():
      trace = dict(
        x=xs,
        y=loss.values * 100,
        type='category',
        hoverinfo='x+y',
        mode='lines',
        line=dict(width=0.5),
        stackgroup='one',
        name=loss_name
      )
      data.append(trace)

    title = "Loss Percentage {}".format(data_name)
    layout = build_base_layout(title)
    overwrite = go.Layout(
      yaxis=dict(
        range=[0, 100],
        dtick=20,
        ticksuffix='%'
      ),
      margin=go.layout.Margin(
       l=40,
      )
    )
    layout.update(overwrite)

    fig = dict(data=data, layout=layout)
    func_name = sys._getframe().f_code.co_name
    win_id = '{} {}'.format(func_name, data_name)
    self.vis.plotlyplot(fig, win=win_id, env=run_name)

    views = ('loss', data_name)
    return views, win_id

  @_add_views
  def loss_line(self, data_dir, run_name, data_name):
    csv_path = data_dir + 'loss.csv'
    df = pd.read_csv(csv_path, index_col='optim_step')

    df = df.sum(axis=1).groupby('optim_step', axis=0)

    std = df.std().values
    mean = df.mean()
    invis = mean.values - std / 2
    xs = mean.index.values

    data = build_loss_trace_with_std(xs, mean, std, invis, stack='one', color='143, 19, 131')

    func_name = sys._getframe().f_code.co_name
    win_id = '{} {}'.format(func_name, data_name)

    title = 'Loss {}'.format(data_name)
    layout = build_base_layout(title)
    overwrite = go.Layout(
      legend=dict(
        y=1.1,
        orientation="h",
        bgcolor='rgba(0,0,0,0)')
    )
    layout.update(overwrite)

    fig = dict(data=data, layout=layout)
    self.vis.plotlyplot(fig, win=win_id, env=run_name)

    views = ('loss', data_name)
    return views, win_id
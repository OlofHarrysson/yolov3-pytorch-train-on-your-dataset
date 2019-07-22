import plotly.graph_objs as go

def x_axis_base():
  return dict(
    titlefont=dict(
      family='Arial, sans-serif',
      size=14,
      color='grey'
    ),        
    title='Optimization Steps',
  )

def build_base_layout(title):
  return go.Layout(
    title=title,
    plot_bgcolor='rgb(229,229,229)',
    xaxis=x_axis_base(),
    margin=go.layout.Margin(
      l=30,
      r=30,
      b=40,
      t=100,
      pad=4
    )
  )

def build_loss_trace_with_std(xs, mean, std, invis_vals, stack, name_prefix='', color=None):
  mean_trace = dict(
    x=xs,
    y=mean.values,
    hoverinfo='x+y',
    type='lines', # Visdom needs this
    mode='lines',
    name='{}Mean'.format(name_prefix),
    line=dict(
      color='rgb({})'.format(color),
    )
  )

  invis_trace = dict(
    x=xs,
    y=invis_vals, # The invisible part underneath std
    type='lines',
    fill='none',
    line=dict(width=0),
    stackgroup=stack,
    showlegend=False,
    hoverinfo='skip'
  )

  std_trace = dict(
    x=xs,
    y=std,
    type='lines',
    line=dict(width=0),
    fillcolor='rgba({},0.2)'.format(color),
    stackgroup=stack,
    name='Std'
  )
  return [mean_trace, invis_trace, std_trace]

def build_max_metrics(max_metrics, legends):
  annotations = []
  y_offset = 0.095

  # Reverse so that annotation order matches legend order
  legends = reversed(legends)
  max_metrics = reversed(max_metrics)

  for i, data in enumerate(zip(max_metrics, legends)):
    text, legend = data
    y = y_offset * i
    annotations.append(dict(
      xref='paper',
      yref='paper',
      text='Max {}: {:.3f}'.format(legend, text),
      showarrow=False,
      x=1,
      y=y,
      clicktoshow='onout',
      bordercolor='#8c8c8c',
      borderwidth=2,
      borderpad=2,
      font=dict(
        family='Courier New, monospace',
        size=14,
      ),
      bgcolor='rgb(220,220,220)',
    ))
  return annotations

def plotly_color_map():
  # From https://stackoverflow.com/a/44727682
  plotly_colors = ['#1f77b4',  # muted blue
                         '#ff7f0e',  # safety orange
                         '#2ca02c',  # cooked asparagus green
                         '#d62728',  # brick red
                         '#9467bd',  # muted purple
                         '#8c564b',  # chestnut brown
                         '#e377c2',  # raspberry yogurt pink
                         '#7f7f7f',  # middle gray
                         '#bcbd22',  # curry yellow-green
                         '#17becf'  # blue-teal
                         ]
  return plotly_colors
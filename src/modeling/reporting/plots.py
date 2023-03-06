from plotnine import *

def make_cv_plot(df, metric_name):
    display = df.copy()
    display.loc[:, 'split_id'] = display.split_id.astype(str)
    p1 = (ggplot(display, aes(x = 'param_id', y = metric_name, color = 'split_id'))
          + geom_point(size = 5)
          + theme_minimal()
          + labs(title = metric_name)
          + scale_y_continuous(expand = (0.2,0))
          + scale_x_continuous(expand = (0.2,0))
         )
    return p1
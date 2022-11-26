from functools import partial
import pandas as pd
import numpy as np

from modeling.models import model_factory
from modeling.validation import splitter_factory, metric_factory

def train_func(df, params, features, target, config):
    model_adapter = model_factory(model_config = config["model"])
    model = model_adapter.train(df = df,
                                params = params["model_params"],
                                features = config["data"]["features"]["asis"],
                                target = config["data"]["target"])
                                         
    return model

def configure_train_func(config):
    
    model_factory(model_config = config["model"])
        
    ret = partial(train_func,
                  features = config["data"]["features"]["asis"],
                  target   = config["data"]["target"],
                  config = config)
                                         
    return ret

def eval_func_template(truth, predictions, metrics):
    scores = {}
    for key in metrics:
        scorer = metric_factory(key)
        scores[key] = [scorer(truth, predictions)]
    ret = pd.DataFrame.from_dict(scores)
            
    return ret

def configure_eval_func(config):
        
    ret = partial(eval_func_template,
                  metrics = config["validation"]["evalulation"]["metrics"])
        
    return ret

def train_and_eval_template(df, params, holdout, config, target):
    
    train_func = configure_train_func(config = config)
    eval_func = configure_eval_func(config = config)
    
    model = train_func(df = df,
                       params = params)
    cutoff = ("cutoff" in config["validation"])
        
    model_adapter = model_factory(model_config = config["model"])
        
    evaldf, predictions = model_adapter.evalulate_result(model = model,
                                                         holdout = holdout,
                                                         target = target,
                                                         task_params = params["model_params"],
                                                         eval_func = eval_func,
                                                         cutoff = cutoff)
            
    return evaldf, predictions

def configure_train_and_eval_func(config):
    
    train_func = configure_train_func(config)
    eval_func = configure_eval_func(config)
        
    train_and_eval_func = partial(train_and_eval_template,
                                  config = config,
                                  target = config["data"]["target"]
                                 )
            
    return train_and_eval_func
    
def split_train_eval_template(df, params, config):
    train = df.iloc[params["splits"]["train"]]
    holdout = df.iloc[params["splits"]["holdout"]]
    
    train_and_eval = configure_train_and_eval_func(config = config)
            
    evaldf, predictions = train_and_eval(df = train,
                                         params = params,
                                         holdout = holdout)
            
    column_order = ["task_id", "param_id", "split_id"] + evaldf.columns.to_list()
    evaldf.loc[:, "task_id"] = params["task_id"]
    evaldf.loc[:, "param_id"] = params["param_id"]
    evaldf.loc[:, "split_id"] = params["split_id"]
    evaldf = evaldf[column_order]
            
    predictions.loc[:, "task_id"] = params["task_id"]
    predictions.loc[:, "param_id"] = params["param_id"]
    predictions.loc[:, "split_id"] = params["split_id"]
                   
    ret = {'eval': evaldf,
           'predictions': predictions
          }
            
    return ret 

def configure_split_train_eval(config):
    split_train_eval = partial(split_train_eval_template,
                               config = config
                              )
    
    return split_train_eval    
    
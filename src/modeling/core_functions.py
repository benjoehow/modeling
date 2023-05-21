from functools import partial
import pandas as pd

from modeling.models import trainer_factory
from modeling.validation import metric_factory
from .config_constants import *


#- Main 
def train_func(df: pd.DataFrame,
               params: dict,
               model_id: dict,
               features, target):

    """

    Parameters
    ----------
    df: pd.DataFrame
        The training data

    params: dict
            The kwargs for the Trainer given by `model_id`

    model_id: str

    features: str

    target: str

    Returns
    -------
    
    """


    trainer = trainer_factory(model_id = model_id)
    model = trainer.train(df = df,
                          params = params,
                          features = features,
                          target = target)
                                         
    return model

def eval_func(truth, predictions, metrics):

    """

    Parameters
    ----------
    truth: 

    predictions:

    metrics: 

    Returns
    -------
    
    """

    scores = {}
    for key in metrics:
        scorer = metric_factory(key)
        if scorer.requires_binary_output:
            #-TODO allow for more cutoff options
            predictions_final = (predictions >= 0.7).astype(int)
        else:
            predictions_final = predictions
        scores[key] = [scorer(truth, predictions_final)]
    ret = pd.DataFrame.from_dict(scores)
            
    return ret

def train_and_eval(df, params, holdout, target, config):
    
    train_func = configure_train_func(config = config)
    eval_func = configure_eval_func(config = config)
    
    model = train_func(df = df, params = params[TASK_PARAM_MODEL_KEY])
        
    evaldf, predictions = model.evalulate_result(holdout = holdout,
                                                 target = target,
                                                 task_params = params[TASK_PARAM_MODEL_KEY],
                                                 eval_func = eval_func)
            
    return evaldf, predictions
    
def split_train_eval(df, params, config):
    train = df.iloc[params["splits"]["train"]]
    holdout = df.iloc[params["splits"]["holdout"]]
    
    train_and_eval = configure_train_and_eval_func(config = config)
            
    evaldf, predictions = train_and_eval(df = train,
                                         params = params,
                                         holdout = holdout)
            
    column_order = [TASK_ID, TASK_PARAM_ID, TASK_SPLIT_ID] + evaldf.columns.to_list()
    evaldf.loc[:, TASK_ID] = params[TASK_ID]
    evaldf.loc[:, TASK_PARAM_ID] = params[TASK_PARAM_ID]
    evaldf.loc[:, TASK_SPLIT_ID] = params[TASK_SPLIT_ID]
    evaldf = evaldf[column_order]
            
    predictions.loc[:, TASK_ID] = params[TASK_ID]
    predictions.loc[:, TASK_PARAM_ID] = params[TASK_PARAM_ID]
    predictions.loc[:, TASK_SPLIT_ID] = params[TASK_SPLIT_ID]
                   
    ret = {'eval': evaldf,
           'predictions': predictions
          }
            
    return ret 

#- Configuration functions
def configure_train_func(config):
    
    ret = partial(train_func,
                  model_id = config[CONFIG_MODEL_KEY][CONFIG_MODEL_ID_KEY],
                  features = config[CONFIG_DATA_KEY][CONFIG_DATA_FEATURES_KEY][CONFIG_DATA_FEATURES_ASIS_KEY],
                  target = config[CONFIG_DATA_KEY][CONFIG_DATA_TARGET_KEY])
                                         
    return ret

def configure_eval_func(config):
        
    ret = partial(eval_func,
                  metrics = config["validation"]["evalulation"]["metrics"])
        
    return ret

def configure_train_and_eval_func(config):
        
    train_and_eval_func = partial(train_and_eval,
                                  config = config,
                                  target = config[CONFIG_DATA_KEY][CONFIG_DATA_TARGET_KEY]
                                 )
            
    return train_and_eval_func

def configure_split_train_eval(config):
    split_train_eval_func = partial(split_train_eval,
                                    config = config
                                    )
    
    return split_train_eval_func   
    
#- Config Validation:

def validate_config(config):

    if CONFIG_MODEL_KEY not in config.keys():
        raise ValueError(f"{CONFIG_MODEL_KEY} not found in top level of config.")
    else:
        if CONFIG_MODEL_ID_KEY not in config[CONFIG_MODEL_KEY].keys():
            raise ValueError(f"{CONFIG_MODEL_ID_KEY} not found in {CONFIG_MODEL_KEY} level of config.")
        elif CONFIG_MODEL_KWARGS_KEY not in config[CONFIG_MODEL_KEY].keys():
            raise ValueError(f"{CONFIG_MODEL_KWARGS_KEY} not found in {CONFIG_MODEL_KEY} level of config.")
        else:
            #-TODO check that kwargs are valid
            for key in config[CONFIG_MODEL_KEY][CONFIG_MODEL_KWARGS_KEY].keys():
                if type(config[CONFIG_MODEL_KEY][CONFIG_MODEL_KWARGS_KEY][key]) == list:
                    if CONFIG_VALIDATION_KEY not in config.keys():
                        raise ValueError("Lists found in model parameters but validation config not found.")
    if CONFIG_DATA_KEY not in config.keys():
        raise ValueError(f"{CONFIG_DATA_KEY} not found in top level of config.")
    else: 
        if CONFIG_DATA_TARGET_KEY not in config[CONFIG_DATA_KEY].keys():
            raise ValueError(f'{CONFIG_DATA_TARGET_KEY} not in data config.')
    if CONFIG_VALIDATION_KEY in config.keys():
        pass
    return True
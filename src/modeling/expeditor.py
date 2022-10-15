from functools import partial
import pandas as pd
import numpy as np
from toolz import pipe
from sklearn.metrics import get_scorer

from modeling.models import model_factory
from modeling.validation import splitter_factory, metric_factory
from .orders import CrossValidationOrder, TrainOrder

class Expeditor():
    
    """
    Configures an Order given a model config and data.
    
    The Expeditor processes the input config to compile 
    the appropriate function and tasks that must be carried
    out by downstream objects. 
    
    ...
    Attributes
    ----------
    config: dict
    
    
    Methods
    -------
    get_order(df)
        
    
    """

    def __init__(self, config: dict):
        
        """
        
        Parameters
        ----------
        config: dict
                The input config 
        """
        
        self.config = config
        self._model_adapter = model_factory(model_config = self.config["model"])
        
    def get_order(self, df):  
        
        """
        
        Parameters
        ----------
        df: dict
            The input configuration 
            
        Returns
        -------
        order: Order
               An order object populated with the neccessary fields to be processed by 
               by downstream objects. 
        """
        
        tasks = self._get_tasks(df = df)
        if "validation" in self.config.keys(): 
            if "splitter" in self.config["validation"].keys():
                func = self._configure_split_train_eval_func()
                order = CrossValidationOrder(config = self.config,
                                             tasks = tasks,
                                             func = func)
        else:
            func = self._configure_train_func()
            order = TrainOrder(config = self.config,
                               tasks = tasks,
                               func = func
                              )
        return order
    
    def _get_tasks(self, df):
        
        tasks = self._compile_tasks(df = df)
        
        return tasks

    def _compile_tasks(self, df):
        
        tasks = self._model_adapter.get_metaparameter_grid(self.config["model"]["params"])
                     
        new_tasks = []
        for task in tasks:
            new_tasks.append({"model_params": task})
        tasks = new_tasks
        
        if "validation" in self.config.keys():
            if "splitter" in self.config["validation"].keys():
                splits = self._get_splits(df = df)
                new_tasks = []
                for s in range(len(splits)):
                    for t in range(len(tasks)):
                        new_task = tasks[t].copy()
                        new_task["splits"] = splits[s]
                        new_task["split_id"] = s
                        new_task["param_id"] = t
                        new_task["task_id"] = (s*len(tasks)) + t
                        new_tasks.append(new_task)
                tasks = new_tasks
        
        return tasks

    def _get_splits(self, df):
        
        split_func  = splitter_factory(self.config["validation"]["splitter"]["id"])
        splitter = split_func(**self.config["validation"]["splitter"]["params"])
        
        splits = []
        for train_index, test_index in splitter.split(df):
            split_indices = {"train": train_index, "holdout": test_index}
            splits.append(split_indices)
            
        return splits
            
    def _configure_train_func(self):
                                                                                                     
        def train_func(df, params, features, target):
            model = self._model_adapter.train(df = df,
                                              params = params["model_params"],
                                              features = features,
                                              target = target)
                                         
            return model
        
        ret = partial(train_func,
                      features = self.config["data"]["features"]["asis"],
                      target = self.config["data"]["target"])
                                         
        return ret
    
    def _configure_eval_func(self):
        
        def eval_func_template(truth, predictions, metrics):
            scores = {}
            for key in metrics:
                scorer = metric_factory(key)
                scores[key] = [scorer(truth, predictions)]
            ret = pd.DataFrame.from_dict(scores)
            
            return ret
        
        ret = partial(eval_func_template,
                      metrics = self.config["validation"]["evalulation"]["metrics"])
        
        return ret
                
    
    def _configure_train_and_eval_func(self):
        
        train_func = self._configure_train_func()
        eval_func = self._configure_eval_func()
        
        def train_and_eval_template(df, params, holdout, target):
            model = train_func(df = df,
                               params = params)
            cutoff = ("cutoff" in self.config["validation"])
            evaldf, predictions = self._model_adapter.evalulate_result(model = model,
                                                                       holdout = holdout,
                                                                       target = target,
                                                                       task_params = params["model_params"],
                                                                       eval_func = eval_func,
                                                                       cutoff = cutoff)
            
            """
            #-TODO: move to adapaters
            predictions_raw = model.predict(df = holdout)
            
            if "cutoff" in self.config["validation"]:
                #-TODO allow for more cutoff options
                predictions = (predictions_raw >= 0.7).astype(int)
            else:
                predictions = predictions_raw
                
            row_id = holdout['row_id'].to_list()
            holdout_list = holdout[target].to_list()

            evaldf = eval_func(truth = holdout_list,
                               predictions = predictions)
            
            predictions = pd.DataFrame({'row_id': row_id,
                                        'predictions': predictions_raw,
                                        'truth': holdout_list})
            #-TODO move to adapters
            """
            
            return evaldf, predictions
        
        train_and_eval_func = partial(train_and_eval_template,
                                      target = self.config["data"]["target"])
            
        return train_and_eval_func
    
    def _configure_split_train_eval_func(self):
        
        train_and_eval = self._configure_train_and_eval_func()
        
        def split_train_eval(df, params):
            train = df.iloc[params["splits"]["train"]]
            holdout = df.iloc[params["splits"]["holdout"]]
            
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
        
        return split_train_eval    
                                     
from functools import partial
import pandas as pd
import numpy as np
from toolz import pipe
from sklearn.metrics import get_scorer

from modeling.models import model_factory
from modeling.validation import splitter_factory
from modeling.validation import metric_factory
from .runner import Runner

class Expeditor():

    def __init__(self, config):
        self.config = config
        self._model_adapter = model_factory(model_config = self.config["model"])
        
    def get_new_runner(self, df):
        
        tasks = self._compile_tasks(df = df)
        runner = Runner(tasks)
        
        return runner
    
    def get_function(self):
        pass

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
                for split in splits:
                    for task in tasks:
                        new_task = task.copy()
                        new_task["splits"] = split
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
                                              params = params,
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
            predictions = model.predict(df = holdout)
            evaldf = eval_func(truth = holdout[target].to_list(),
                               predictions = predictions)
            
            return evaldf
        
        train_and_eval_func = partial(train_and_eval_template,
                                      target = self.config["data"]["target"])
            
        return train_and_eval_func
    
    def _configure_split_train_eval_func(self):
        
        train_and_eval = self._configure_train_and_eval_func()
        
        def split_train_eval(df, params):
            train = df.iloc[params["splits"]["train"]]
            holdout = df.iloc[params["splits"]["holdout"]]
            evaldf = train_and_eval(df = train,
                                    params = params["model_params"],
                                    holdout = holdout)
                         
            return evaldf
        
        return split_train_eval    
                                     
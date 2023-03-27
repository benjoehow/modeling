from abc import ABC, abstractmethod
import pandas as pd
import logging
import uuid

from modeling.models import model_factory
from modeling.validation import splitter_factory, metric_factory
from .core_functions import *

class Order(ABC):
    
    """
    Abstract Base Class for orders. 
    
    Orders contain:
     - A function to apply to a pandas.DataFrame
     - A list of tasks (parameters to the above function)
     - Results by applying those tasks to data
     - A copy of the original config
     
    ...
    Attributes
    ----------
    df: 
    is_finished: boolean
    completed_tasks: 
    config: dict
    tasks: list of dict
    func: function
    
    Methods
    -------
    get_tasks()
    add_result(result)
    get_results()
   
    """
    
    def __init__(self, df, config, tasks, func,
                 order_id = str(uuid.uuid4())):
        
        self.order_id = order_id
        self.df = df
        self.is_finished = False
        self.completed_task_count = 0
        self.completed_tasks = None
        self.config = config
        self.tasks = tasks
        self.total_tasks = len(self.tasks)
        self.func = func
    
    @abstractmethod
    def get_tasks(self):
        return NotImplemented
    
    def add_result(self, result):
        self.completed_task_count += 1
        self._store_result(result)
        logging.info(f'Order {self.order_id} - adding result. ' + 
                     f'{self.completed_task_count} / {self.total_tasks} tasks completed.')
    
    @abstractmethod
    def _store_result(self, result):
        return NotImplemented
    
    def get_results(self):
        return self.completed_tasks
    
    
class CrossValidationOrder(Order):
    
    """
    Cross Validation Order
    
    Cross Validation tasks comprise of 
    - metaparameters for the model
    - indicies to split the given df on (for train and test sets)
    Completed tasks are stored in a dictionary with keys
    - 'eval': a pandas.DataFrame containing the metaparameters
              and evaluation metric results, indexed by task_id
    - 'predictions': the prediction and ground truth from the holdout
                     dataset, indexed by task_id
     
    ...
    Attributes
    ----------
    is_finished: boolean
    completed_tasks: 
    config: dict
    tasks: list of dict
    func: function
    
    Methods
    -------
    get_tasks()
    add_result(result: dict)
    get_results()
   
    """
    
    def __init__(self, df, config, tasks, func):
        super().__init__(df = df,
                         config = config,
                         tasks = tasks,
                         func = func)
        
        self.completed_tasks = {'eval': pd.DataFrame(),
                                'predictions': pd.DataFrame()
                               }
        
    def get_tasks(self):
        return self.tasks
    
    def _store_result(self, result):
        
        self.completed_tasks['eval'] = pd.concat([self.completed_tasks['eval'],
                                                  result['eval']])
        self.completed_tasks['predictions'] = pd.concat([self.completed_tasks['predictions'], 
                                                         result['predictions']])
        
        if(len(self.tasks) == self.completed_tasks['eval'].task_id.nunique()):
            self.is_finished = True
            
    def get_results(self):
        ret = self.completed_tasks.copy()
        ret['eval'] = ret['eval'].reset_index().to_dict()
        ret['predictions'] = ret['predictions'].reset_index().to_dict()
        return ret
            
class TrainOrder(Order):
    
    def __init__(self, df, config, tasks, func):
        super().__init__(df = df, 
                         config = config,
                         tasks = tasks,
                         func = func)
        
        self.completed_tasks = None
        
    def get_tasks(self):
        return self.tasks
    
    def _store_result(self, result):
        self.completed_tasks = result
        self.is_finished = True
        
        
def get_order(df, config):  
    
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
    
    tasks = _compile_tasks(df = df, config = config)
    if "validation" in config.keys(): 
        if "splitter" in config["validation"].keys():
            func = configure_split_train_eval(config = config)
            order = CrossValidationOrder(df = df,
                                         config = config,
                                         tasks = tasks,
                                         func = func)
    else:
        func = configure_train_func(config = config)
        order = TrainOrder(df = df, 
                           config = config,
                           tasks = tasks,
                           func = func
                          )
    return order

def _compile_tasks(df, config):
    
    model_adapter = model_factory(model_config = config["model"])
    
    tasks = model_adapter.get_metaparameter_grid(config["model"]["params"])
                    
    new_tasks = []
    for task in tasks:
        new_tasks.append({"model_params": task})
    tasks = new_tasks
    
    if "validation" in config.keys():
        if "splitter" in config["validation"].keys():
            splits = _get_splits(df = df, config = config)
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

def _get_splits(df, config):
    
    split_func  = splitter_factory(config["validation"]["splitter"]["id"])
    splitter = split_func(**config["validation"]["splitter"]["params"])
    
    splits = []
    for train_index, test_index in splitter.split(df):
        split_indices = {"train": train_index, "holdout": test_index}
        splits.append(split_indices)
        
    return splits                                 
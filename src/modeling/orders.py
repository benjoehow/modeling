from abc import ABC, abstractmethod
import pandas as pd

class Order(ABC):
    
    """
    Abstract Base Class for orders. 
    
    Orders contain:
     - A function to apply to a pandas.DataFrame
     - A list of tasks (parameters to the above function)
     - A 
    
    ...
    Attributes
    ----------
    order: Order
    
    Methods
    
    
    """
    
    def __init__(self, config, tasks, func):
        self.is_finished = False
        self.completed_tasks = None
        self.config = config
        self.tasks = tasks
        self.func = func
        
    def get_tasks(self):
        return NotImplemented
    
    def add_result(self, result):
        return NotImplemented
    
    def get_results(self):
        return self.completed_tasks
    
class CrossValidationOrder(Order):
    
    def __init__(self, config, tasks, func):
        super().__init__(config = config, tasks = tasks, func = func)
        self.completed_tasks = {'eval': pd.DataFrame(),
                                'predictions': pd.DataFrame()
                               }
        
    def get_tasks(self):
        return self.tasks
    
    def add_result(self, result):
        
        self.completed_tasks['eval'] = pd.concat([self.completed_tasks['eval'],
                                            result['eval']])
        self.completed_tasks['predictions'] = pd.concat([self.completed_tasks['predictions'], 
                                                   result['predictions']])
        if(len(self.tasks) == len(self.completed_tasks['eval'])):
            self.is_finished = True
            
class TrainOrder(Order):
    
    def __init__(self, config, tasks, func):
        super().__init__(config = config, tasks = tasks, func = func)
        self.completed_tasks = None
        
    def get_tasks(self):
        return self.tasks
    
    def add_result(self, result):
        self.completed = result
        self.is_finished = True
from abc import ABC, abstractmethod
import pandas as pd

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
        if(len(self.tasks) == self.completed_tasks['eval'].task_id.nunique()):
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
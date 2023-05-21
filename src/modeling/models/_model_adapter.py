from abc import ABC, abstractmethod

##-TODO split out validation from training

class Trainer(ABC):
    
    _skip_job_keys = []
    
    def get_metaparameter_grid(self, params):
        
        grid = [params]
        
        for key in params:
            if key in self._skip_job_keys:
                continue
            if isinstance(params[key], list):
                new_grid = []
                for grid_slice in grid:
                    current_params = self._expand_params(params = grid_slice, key = key)
                    new_grid.extend(current_params)   
                grid = new_grid
                                    
        return grid
    
    def _expand_params(self, params, key):
        
        ret = []
        
        for value in params[key]:
            params_copy = params.copy()
            params_copy[key] = value
            ret.append(params_copy)
        
        return ret
    
    @abstractmethod
    def prep_data(self, df, label, features):
        return NotImplemented
    
    @abstractmethod
    def train(self, features, target):
        return NotImplemented
    

class Predictor(ABC):
    
    def __init__(self, model, features):
        self._model = model
        self._features = features
    
    @abstractmethod
    def get_post_train_diagnostics(self, model):
        return NotImplemented
    
    @abstractmethod
    def predict(self, data):
        return NotImplemented
    
    @abstractmethod
    def evalulate_result(self, model, holdout, task_params, eval_params):
        return NotImplemented
    
    
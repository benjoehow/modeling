from abc import ABC, abstractmethod

class model_adapter(ABC):
    
    _skip_job_keys = []
    
    def __init__(self, model_config):
        super.__init__(model_config = config)
    
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
    
    def prep_data(self, df, label, features):
        return NotImplemented
    
    @abstractmethod
    def train(self, features, target):
        return NotImplemented
    
    @abstractmethod
    def get_post_train_diagnostics(self, model):
        return NotImplemented
        
    def evalulate_result(self, model, holdout, task_params, eval_params):
        return NotImplemented
    
    
class model_wrapper(ABC):
    
    def __init__(self, model, features):
        self._model = model
        self._features = features
    
    @abstractmethod
    def predict(self, data):
        return NotImplemented
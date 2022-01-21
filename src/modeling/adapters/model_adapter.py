from abc import ABC, abstractmethod

class model_adapter(ABC):
    
    _skip_job_keys = []
    
    def __init__(self):
        pass
    
    def get_jobs(self, params):
        
        ret = [params]
        
        for key in params:
            if key in self._skip_job_keys:
                continue
            if isinstance(params[key], list):
                new_jobs = []
                for job in ret:
                    new_jobs.extend(self._expand_params(params = job, key = key))
                ret = new_jobs
                                    
        return ret
    
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
    def train(self):
        return NotImplemented
    
    @abstractmethod
    def get_post_train_diagnostics(self, model):
        return NotImplemented
    
    @abstractmethod
    def predict(self, params, data):
        return NotImplemented
        
    def evalulate_result(self, target_vector, predicted_vector, job_params, eval_params):
        return NotImplemented
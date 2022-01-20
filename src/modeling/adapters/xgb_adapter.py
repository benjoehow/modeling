import xgboost as xgb


class xgb_adapter:
    
    def __init__(self):
        pass
    
    def get_jobs(self, params):
        jobs = [params]
        
        for key in params:
            if key == "num_boost_round":
                continue
            if isinstance(params[key], list):
                new_jobs = []
                for job in jobs:
                    new_jobs.extend(self._expand_params(params = job, key = key))
                jobs = new_jobs
                                    
        return jobs
                
    def _expand_params(self, params, key):
        
        ret = []
        
        for value in params[key]:
            params_copy = params.copy()
            params_copy[key] = value
            ret.append(params_copy)
        
        return ret
          
    def prep_data(self, df, label, features):
        ret = xgb.DMatrix(data = df[features],
                          label = df[label],
                          feature_names = features)
        return ret 
        
    def train(self, params, data):
        
        params_run = params.copy()
        num_boost_round = max(params_run["num_boost_round"])
        params_run.pop("num_boost_round")
        
        ret = xgb.train(params = params_run,
                          dtrain = data,
                          num_boost_round = num_boost_round)
        return ret
        
    #def predict(self, model, data):
        
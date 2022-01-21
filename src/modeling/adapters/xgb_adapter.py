import xgboost as xgb
from model_adapter import model_adapter

class xgb_adapter(model_adapter):
    
    _skip_job_keys = ["num_boost_round"]
    
    def __init__(self):
        pass
                
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
        
    def get_post_train_diagnostics(self, model):
        pass
        
    def predict(self, model, data):
        pass
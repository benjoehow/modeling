import xgboost as xgb
from modeling.models import model_adapter, model_wrapper
from modeling import proba_cutoff

class xgboost_adapter(model_adapter):
    
    _skip_job_keys = ["num_boost_round"]
    
    def __init__(self, model_config):
        self.model_config = model_config
                
    def _expand_params(self, params, key):
        
        ret = []
        
        for value in params[key]:
            params_copy = params.copy()
            params_copy[key] = value
            ret.append(params_copy)
        
        return ret
        
    def train(self, params, df, features, target):
        
        data = xgb.DMatrix(data = df[features],
                           label = df[target],
                           feature_names = features)
        
        params_run = params.copy()
        num_boost_round = max(params_run["num_boost_round"])
        params_run.pop("num_boost_round")
        
        model = xgb.train(params = params_run,
                          dtrain = data,
                          num_boost_round = num_boost_round)
        
        ret = xgboost_wrapper(model = model,
                              features = features)
        
        return ret
        
    def get_post_train_diagnostics(self, model):
        pass
        
    

class xgboost_wrapper(model_wrapper):
    
    def __init__(self, model, features):
        self._model = model
        self._features = features
        
    def predict(self, df):
        
        data = xgb.DMatrix(data = df[self._features],
                           feature_names = self._features)
        
        predictions = self._model.predict(data)
        
        predictions = (predictions >= 0.7).astype(int)
        return predictions
        
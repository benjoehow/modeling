import xgboost as xgb
from modeling.models import Trainer, Predictor

import pandas as pd

from ..config_constants import *

class XGBoostTrainer(Trainer):
    
    _skip_job_keys = ["num_boost_round"]

    def prep_data(self,
                  df: pd.DataFrame,
                  features: list[str],
                  target: str) -> xgb.DMatrix:

        data = xgb.DMatrix(data = df[features],
                           label = df[target],
                           feature_names = features)
        
        return data
        
    def train(self, params, df, features, target):
        #print(f"train keys: {params.keys()}")
        data = self.prep_data(df = df,
                              features = features,
                              target = target)
        params_run = params.copy()
        if type(params_run["num_boost_round"]) == list:
            num_boost_round = max(params_run["num_boost_round"])
        else:
            num_boost_round = params_run["num_boost_round"]
        
        params_run.pop("num_boost_round")
        
        model = xgb.train(params = params_run,
                          dtrain = data,
                          num_boost_round = num_boost_round)
        
        ret = XGBoostPredictor(model = model,
                               features = features)
        
        return ret
    
    
  
    

class XGBoostPredictor(Predictor):
    
    def __init__(self, model, features):
        self._model = model
        self._features = features
        
    def predict(self, df, num_boost_round_eval = 0):
        
        data = xgb.DMatrix(data = df[self._features],
                           feature_names = self._features)
        
        predictions = self._model.predict(data,
                                          iteration_range=(0, num_boost_round_eval))
        
        return predictions
    
    def get_post_train_diagnostics(self, model):
        pass
    
    def evalulate_result(self, 
                         holdout,
                         target,
                         task_params,
                         eval_func):
        
        if TASK_PARAM_MODEL_KEY in task_params.keys():
            raise KeyError(f"{TASK_PARAM_MODEL_KEY} found in params argument in Predictor's evalulate_result")
        
        evaldf_final = pd.DataFrame()
        predictions_final = pd.DataFrame()
        
        num_boost_rounds = task_params["num_boost_round"]
        #-TODO: Generalize
        if type(num_boost_rounds) == int:
            num_boost_rounds = [num_boost_rounds]
            
        for num_boost_round in num_boost_rounds:
            predictions_raw = self.predict(df = holdout,
                                           num_boost_round_eval = num_boost_round)
                
            row_id = holdout['row_id'].to_list()
            holdout_list = holdout[target].to_list()

            evaldf = eval_func(truth = holdout_list,
                               predictions = predictions_raw)
            
            evaldf['num_boost_round'] = num_boost_round
            for key in task_params.keys():
                if key != 'num_boost_round':
                    evaldf[key] = task_params[key]
            
            predictions = pd.DataFrame({'row_id': row_id,
                                        'predictions': predictions_raw,
                                        'truth': holdout_list})
            
            predictions['num_boost_round'] = num_boost_round
            for key in task_params.keys():
                if key != 'num_boost_round':
                    predictions[key] = task_params[key]
            
            evaldf_final = pd.concat([evaldf, evaldf_final])
            predictions_final = pd.concat([predictions, predictions_final])
                    
        return evaldf_final, predictions_final
        
        
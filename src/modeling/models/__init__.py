
from ._model_adapter import Trainer, Predictor
from ._xgb_adapter import XGBoostTrainer

_trainer_lookup = {}
_trainer_lookup["xgboost"] = XGBoostTrainer

def trainer_factory(model_id):
    
    if model_id not in _trainer_lookup.keys():
        raise ValueError(f"Model key: {model_id} not found")
        
    return _trainer_lookup[model_id]()

__all__ = [
    "trainer_factory"
    "Trainer",
    "Predictor"
]
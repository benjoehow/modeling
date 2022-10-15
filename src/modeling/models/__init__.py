
from ._model_adapter import model_adapter, model_wrapper
from ._xgb_adapter import xgboost_adapter

_adapter_lookup = {}
_adapter_lookup["xgboost"] = xgboost_adapter

def model_factory(model_config):
    
    key = model_config["id"]
    
    if key not in _adapter_lookup.keys():
        raise ValueError("Model key not found")
        
    return _adapter_lookup[key](model_config = model_config)

__all__ = [
    "model_adapter",
    "model_factory"
    "model_wrapper"
]
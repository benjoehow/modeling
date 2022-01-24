
from ._model_adapter import model_adapter
from ._xgb_adapter import xgboost_adapter

_adapter_lookup = {}
_adapter_lookup["xgboost"] = xgboost_adapter

def model_factory(key):
    
    if key not in _adapter_lookup.keys():
        raise ValueError("Model key not found")
        
    return _adapter_lookup[key]()

__all__ = [
    "model_adapter",
    "model_factory"
]
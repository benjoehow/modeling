import pytest
import json
from pathlib import Path
from sklearn.datasets import load_wine

from modeling.models import Predictor
import modeling 

TEST_DATA_DIR = Path(__file__).resolve().parent / 'data'

def get_wine_df():
    wine = load_wine(as_frame = True).frame
    wine["target2"] = (wine.target==1).astype(int).astype(int)
    wine.loc[:,"row_id"] = wine.index
    return wine

def load_config(filename):
    with open(filename, 'r') as config_file:
        config = json.load(config_file)
    return config
 
wine = get_wine_df()
config = load_config(filename = f'{TEST_DATA_DIR}/cv_reg_config.json')
order = modeling.get_order(df = wine, config = config)

#order.tasks[0]['model_params']
train_func = modeling.configure_train_func(config = order.config)
m = train_func(df = wine, params = order.tasks[0]['model_params'])

def test_that_config_is_valid():
    assert modeling.validate_config(config = config) == True

def test_that_train_func_returns_Predictor():
    assert isinstance(m, Predictor) == True
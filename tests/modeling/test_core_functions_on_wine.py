import pytest
import json
from pathlib import Path
from sklearn.datasets import load_wine

from modeling.models import Predictor
import modeling 

TEST_DATA_DIR = Path(__file__).resolve().parent / 'data'

@pytest.fixture()
def wine_df():
    wine = load_wine(as_frame = True).frame
    wine["target2"] = (wine.target==1).astype(int).astype(int)
    wine.loc[:,"row_id"] = wine.index
    return wine

@pytest.fixture(params=['reg_config', 'cv_reg_config'])
def config(request):
    filename = request.param
    with open(f'{TEST_DATA_DIR}/{filename}.json', 'r') as config_file:
        ret = json.load(config_file)
    return ret
 
@pytest.fixture()
def order(wine_df, config):
    ret = modeling.get_order(df = wine_df, config = config)
    return ret

def test_that_config_is_valid(config):
    assert modeling.validate_config(config = config) == True

def test_that_train_func_returns_Predictor(order):
    train_func = modeling.configure_train_func(config = order.config)
    m = train_func(df = order.df, params = order.tasks[0])
    assert isinstance(m, Predictor) == True


def test_that_runner_executes_the_order(order):
    runner = modeling.Runner(workers = 4)
    runner.run(order = order)

    assert order.is_finished == True


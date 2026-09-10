import pytest
import numpy as np
import sklearn
from sklearn.metrics import root_mean_squared_error
from assg.tasks import task1_load_data
from assg.tasks import task5_lasso_model


@pytest.fixture
def data():
    x, y = task1_load_data()
    model = task5_lasso_model(x, y)
    return x, y, model

def test_model_pipeline(data):
    _, _, model = data
    assert isinstance(model, sklearn.pipeline.Pipeline)
    assert isinstance(model[0], sklearn.preprocessing._polynomial.PolynomialFeatures)
    pf = model[0]
    assert pf.degree == 100
    pf_params = {'degree': 100, 'include_bias': False, 'interaction_only': False, 'order': 'C'}
    assert pf.get_params() == pf_params
    assert isinstance(model[1], sklearn.linear_model._coordinate_descent.Lasso)
    params = model[1].get_params()
    assert 'alpha' in params.keys()
    
def test_coef(data):
    _, _, model = data
    coef = model[1].coef_
    assert len(coef) == 100

def test_r2score(data):
    x, y, model = data
    r2score = model.score(x, y)
    assert r2score > 0.9

def test_rmse(data):
    x, y, model = data
    y_predict = model.predict(x)
    rmse = root_mean_squared_error(y, y_predict)
    assert rmse < 0.2

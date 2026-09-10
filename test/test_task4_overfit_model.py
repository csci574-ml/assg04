import pytest
import numpy as np
import sklearn
from sklearn.metrics import root_mean_squared_error
from assg.tasks import task1_load_data
from assg.tasks import task4_overfit_model


@pytest.fixture
def data():
    x, y = task1_load_data()
    model = task4_overfit_model(x, y)
    return x, y, model

def test_model_pipeline(data):
    _, _, model = data
    assert isinstance(model, sklearn.pipeline.Pipeline)
    assert isinstance(model[0], sklearn.preprocessing._polynomial.PolynomialFeatures)
    pf = model[0]
    assert pf.degree == 100
    pf_params = {'degree': 100, 'include_bias': False, 'interaction_only': False, 'order': 'C'}
    assert pf.get_params() == pf_params
    assert isinstance(model[1], sklearn.linear_model._base.LinearRegression)
    lr = model[1]
    lr_params = {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'positive': False, 'tol': 1e-06}
    assert lr.get_params() == lr_params
    
def test_intercept(data):
    _, _, model = data
    intercept = model[1].intercept_
    assert intercept == pytest.approx(1.9238693956195079, rel=1e-1)

def test_coef(data):
    _, _, model = data
    coef = model[1].coef_
    assert len(coef) == 100
    assert coef[0] == pytest.approx(2.201309741610785, rel=1e-1)
    assert coef[1] == pytest.approx(2.12812005853831, rel=1e-1)
    assert coef[10] == pytest.approx(-2003.2210346900065, rel=1e-1)
    assert coef[50] == pytest.approx(-402.4015806228064, rel=1e-1)
    assert coef[99] == pytest.approx(-2188.476258492196, rel=1e-1)

def test_r2score(data):
    x, y, model = data
    r2score = model.score(x, y)
    assert r2score == pytest.approx(0.9973959804305983, rel=1e-2)

def test_rmse(data):
    x, y, model = data
    y_predict = model.predict(x)
    rmse = root_mean_squared_error(y, y_predict)
    assert rmse == pytest.approx(0.04106771555055683, rel=1e-2)

import pytest
import numpy as np
import sklearn
from sklearn.metrics import root_mean_squared_error
from assg.tasks import task1_load_data
from assg.tasks import task2_underfit_model


@pytest.fixture
def data():
    x, y = task1_load_data()
    model = task2_underfit_model(x, y)
    return x, y, model

def test_model_pipeline(data):
    _, _, model = data
    assert isinstance(model, sklearn.pipeline.Pipeline)
    assert isinstance(model[0], sklearn.preprocessing._polynomial.PolynomialFeatures)
    pf = model[0]
    assert pf.degree == 2;
    pf_params = {'degree': 2, 'include_bias': False, 'interaction_only': False, 'order': 'C'}
    assert pf.get_params() == pf_params
    assert isinstance(model[1], sklearn.linear_model._base.LinearRegression)
    lr = model[1]
    lr_params = {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'positive': False, 'tol': 1e-06}
    assert lr.get_params() == lr_params
    
def test_intercept(data):
    _, _, model = data
    intercept = model[1].intercept_
    assert intercept == pytest.approx(2.086873037695686, rel=1e-4)

def test_coef(data):
    _, _, model = data
    coef = model[1].coef_
    assert coef[0] == pytest.approx(0.81875687, rel=1e-4)
    assert coef[1] == pytest.approx(-1.36223923, rel=1e-4)

def test_r2score(data):
    x, y, model = data
    r2score = model.score(x, y)
    assert r2score == pytest.approx(0.7769591377499175, rel=1e-4)

def test_rmse(data):
    x, y, model = data
    y_predict = model.predict(x)
    rmse = root_mean_squared_error(y, y_predict)
    assert rmse == pytest.approx(0.31486435395552376, rel=1e-4)


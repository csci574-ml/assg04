import pytest
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from assg.tasks import learning_curve_errors


@pytest.fixture
def data():
    np.random.seed(42)
    x = np.linspace(-5.0, 5.0, 100)
    y = 2.5 * x + 1.0 + np.random.randn(100) * 1.0
    x = x.reshape(-1, 1)
    model = LinearRegression()
    train_errors, test_errors = learning_curve_errors(model, x, y)
    return train_errors, test_errors

def test_train_errors(data):
    train_errors, _ = data
    assert len(train_errors) == 79
    expected_errors = np.array([0.0, 1.3710242980056706e-15, 0.008226499792554355, 0.6060453617587794, 0.5519805145155924])
    assert np.allclose(train_errors[:5], expected_errors)
    assert max(train_errors) ==  pytest.approx(1.0929159457839905)

def test_test_errors(data):
    _, test_errors = data
    assert len(test_errors) == 79
    expected_errors = np.array([6.807111779574341, 2.8618559336108844, 2.8588563844249277, 1.8483844530375642, 1.9383035157683932])
    assert np.allclose(test_errors[:5], expected_errors)
    assert max(test_errors) == pytest.approx(6.807111779574341)


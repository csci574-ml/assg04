import pytest
import numpy as np
from assg.tasks import task1_load_data


@pytest.fixture
def data():
    x, y = task1_load_data()
    return x, y

def test_loaded_types(data):
    x, y = data
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)

def test_x_properties(data):
    x, _ = data
    assert x.shape == (100, 1)

def test_y_properties(data):
    _, y = data
    assert y.shape == (100,)

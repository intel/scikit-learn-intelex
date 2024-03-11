import numpy as np
import numpy.random as rand

import pytest

from sklearnex.utils import _assert_all_finite

@pytest.mark.parameterize("dtype",[np.float32, np.float64])
@pytest.mark.parameterize("shape",[[16,2048],[2**16+3,],[1000,1000])
@pytest.mark.parameterize("allow_nan",[False, True])
def test_sum_infinite_actually_finite(dtype, shape, allow_nan):
  X = np.array(shape)
  X.fill(np.finfo(dtype).max)
  _assert_all_finite(X, allow_nan=allow_nan)

@pytest.mark.parameterize("dtype",[np.float32, np.float64])
@pytest.mark.parameterize("shape",[[16,2048],[2**16+3,],[1000,1000])
@pytest.mark.parameterize("allow_nan",[False, True])
def test_assert_finite_random_location(dtype, shape, allow_nan):
  seed = int(time.time())
  rand.seed(seed)
  pass

@pytest.mark.parameterize("dtype",[np.float32, np.float64])
@pytest.mark.parameterize("allow_nan",[False, True])
def test_assert_finite_random_shape_and_location(dtype, allow_nan):
  seed = int(time.time())
  rand.seed(seed)
  pass


  pass



import numpy as np

def add_intercept(x: np.ndarray, axis: int = 1) -> np.ndarray:
  """Adds a column of 1's to the non-empty numpy.ndarray x.
  Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
  Returns:
    X as a numpy.ndarray, a vector of dimension m * 2.
    None if x is not a numpy.ndarray.
    None if x is a empty numpy.ndarray.
  Raises:
    This function should not raise any Exception.
  """
  ones = np.ones((x.shape[0], 1))
  res = np.concatenate((ones, x), axis=axis)
  return res

def predict(thetas, x: np.ndarray) -> np.ndarray:
  """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
  Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
  Returns:
    y_hat as a numpy.ndarray, a vector of dimension m * 1.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
  Raises:
    This function should not raise any Exception.
  """
  theta = thetas
  intercepted = add_intercept(x)
  if intercepted.shape[1] != theta.shape[0]:
    return None
  y_hat = intercepted.dot(theta)
  return y_hat

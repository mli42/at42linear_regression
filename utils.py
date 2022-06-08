import numpy as np

def add_intercept(x: np.ndarray) -> np.ndarray:
    """Adds a column of 1's on the left to the non-empty numpy.ndarray x.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
    Returns:
        x as a numpy.ndarray, a vector of dimension m * 2.
    """
    ones = np.ones((x.shape[0], 1))
    res = np.concatenate((ones, x), axis=1)
    return res

def predict(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None on error
    """
    x_prime = add_intercept(x)
    if (x_prime is None or
        x_prime.shape[1] != theta.shape[0] or
        theta.shape[1] != 1):
        return None
    y_hat = x_prime.dot(theta)
    return y_hat

def get_theta() -> np.ndarray:
    theta = np.asarray([[0],[0]])
    return theta

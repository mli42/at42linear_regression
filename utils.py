import numpy as np

THETA_FILE = "theta.txt"

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

def str_array(array: np.ndarray) -> str:
    return str(array).replace('\n', '')

def save_theta(theta: np.ndarray) -> None:
    print(f"{save_theta.__name__}: saving {str_array(theta)}")
    try:
        with open(THETA_FILE, "w") as file:
            for element in theta:
                file.write(f"{element[0]}\n")
    except Exception as e:
        print(f"{save_theta.__name__} failed: {e}")

def get_theta() -> np.ndarray:
    theta = np.asarray([[.0],[.0]])

    try:
        tmp_theta = theta.copy()
        with open(THETA_FILE, "r") as file:
            for i, line in enumerate(file):
                if i >= 2:
                    raise Exception("File too long")
                tmp_theta[i][0] = float(line.strip())
                if np.isnan(tmp_theta[i][0]):
                    raise Exception("Has NaN")
            if i < 1:
                raise Exception("File too short")
        theta = tmp_theta
    except Exception as e:
        print(f"File `{THETA_FILE}` corrupted: {e}")
        save_theta(theta)
    return theta

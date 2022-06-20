import numpy as np
import pandas as pd
from typing import Tuple

THETA_FILE = "theta.txt"

def predict(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None on error
    """
    y_hat = theta[0][0] + theta[1][0] * x
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

def minmax(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
    Args:
        x: has to be an numpy.ndarray, m * 1.
    Returns:
        x' as a numpy.ndarray, m * 1.
    """
    span = x_max - x_min
    res = (x - x_min) / span
    return res

def get_data() -> Tuple[np.ndarray, np.ndarray]:
    """Read the data.csv and returns the data

    Returns:
        Tuple[np.ndarray, np.ndarray]: km (x), price (y)
        None on error
    """
    try:
        data = pd.read_csv("./resources/data.csv")
    except Exception as e:
        print(f"{get_data.__name__}: {e}")
        exit(1)
    km = data["km"].values.reshape(-1, 1)
    price = data["price"].values.reshape(-1, 1)
    return (km, price)

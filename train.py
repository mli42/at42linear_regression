#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import utils
import argparse
from typing import List

class MyLinearRegression():

    def __init__(self, alpha: float = 0.001, max_iter: int = 1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = utils.get_theta()
        print(f"MyLR: Using {self.alpha = }, and {self.max_iter = }")
        print(f"MyLR: got theta {utils.str_array(self.theta)}")

    @staticmethod
    def cost_(y: np.ndarray, y_hat: np.ndarray) -> float:
        """Computes the mean squared error of two non-empty numpy.ndarray.
            The two arrays must have the same dimensions.
        Args:
            y: has to be an numpy.ndarray, a vector.
            y_hat: has to be an numpy.ndarray, a vector.
        Returns:
            The mean squared error (MSE) of the two vectors as a float.
        """
        if y.shape != y_hat.shape:
            return None
        j_elem = (y_hat - y) ** 2 / y.shape[0]
        return np.sum(j_elem)

    def plot_hypo(self, x: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> None:
        plt.figure()
        # Data repartition
        plt.plot(x, y, "o")
        # Prediction model
        plt.plot(x, y_hat)
        plt.title('Data repartition and predition model')
        plt.legend(['Dataset','Hypothesis'])
        plt.xlabel("Mileage (km)")
        plt.ylabel("Price of car")

    def plot(self, x: np.ndarray, y: np.ndarray, y_hat: np.ndarray, losses: List[float]) -> None:
        self.plot_hypo(x, y, y_hat)

        plt.figure()
        plt.title('Train loss through epochs')
        plt.plot(losses)
        plt.legend(['Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.show()

    def minmax(self, x: np.ndarray) -> np.ndarray:
        """Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
        Args:
            x: has to be an numpy.ndarray, m * 1.
        Returns:
            x' as a numpy.ndarray, m * 1.
        """
        res = utils.minmax(x, np.min(x), np.max(x))
        return res

    def gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Computes a gradient vector from three non-empty numpy.ndarray
        Args:
            x: has to be a numpy.ndarray, a matrix of dimension m * 1.
            y: has to be a numpy.ndarray, a vector of dimension m * 1.
            theta: has to be a numpy.ndarray, a 2 * 1 vector.
        Returns:
            The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
        """
        m = x.shape[0]
        theta0 = self.alpha * np.sum(utils.predict(x, self.theta) - y) / m
        theta1 = self.alpha * np.sum((utils.predict(x, self.theta) - y) * x) / m
        nabla_j = np.asarray([[theta0], [theta1]])
        return nabla_j

    def fit(self, x: np.ndarray, y: np.ndarray, show_gradient: bool) -> None:
        """
        Description:
            Fits the model to the training dataset contained in x and y.
        Args:
            x: a vector of dimension m * 1
            y: a vector of dimension m * 1
            theta: a vector of dimension 2 * 1.
            alpha: a float, the learning rate
            max_iter: an int, the number of iterations done
        """
        alpha = self.alpha
        if x.shape != y.shape or self.theta.shape != (2, 1) or self.max_iter <= 0:
            return None
        norm_x = self.minmax(x)
        losses = []
        for i in range(self.max_iter):
            gradient = self.gradient(norm_x, y)
            self.theta -= gradient

            y_hat = utils.predict(norm_x, self.theta)
            running_loss = MyLinearRegression.cost_(y, y_hat)
            losses.append(running_loss)

            if show_gradient and i % 100 == 0:
                self.plot_hypo(x, y, y_hat)
                plt.show()

        utils.save_theta(self.theta)
        print(f"Last loss: {losses[-1]}")

        self.plot(x, y, y_hat, losses)

def main():
    parser = argparse.ArgumentParser(description='Train model with linear regression')
    parser.add_argument('--alpha', action='store', default=0.1, type=float,
        help='define learning rate (default: 0.1)')
    parser.add_argument('--max_iter', action='store', default=1000, type=int,
        help='define number of iterations (default: 1000)')
    parser.add_argument('--show', action='store_true',
        help='display plots during gradient descent')
    args = parser.parse_args()

    mylr = MyLinearRegression(alpha=args.alpha, max_iter=args.max_iter)
    km, price = utils.get_data()
    mylr.fit(km, price, args.show)

if __name__ == "__main__":
    main()

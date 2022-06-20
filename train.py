import numpy as np
import matplotlib.pyplot as plt
import utils

class MyLinearRegression():

    def __init__(self, alpha: float = 0.001, max_iter: int = 1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = utils.get_theta()
        print(f"MyLR: got theta {utils.str_array(self.theta)}")

    @staticmethod
    def mse_(y: np.ndarray, y_hat: np.ndarray) -> float:
        if y.shape != y_hat.shape:
            return None
        mse_elem = (y_hat - y) ** 2 / (y.shape[0])
        return np.sum(mse_elem)

    @staticmethod
    def cost_elem_(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """
        Description:
            Calculates all the elements (1/2*M)*(y_pred - y)^2 of the cost function.
        Args:
            y: has to be an numpy.ndarray, a vector.
            y_hat: has to be an numpy.ndarray, a vector.
        Returns:
            J_elem: numpy.ndarray, a vector of dimension (number of the training examples,1).
            None if there is a dimension matching problem between X, Y or theta.
        Raises:
            This function should not raise any Exception.
        """
        if y.shape != y_hat.shape:
            return None
        res = (y_hat - y) ** 2 / (2 * y.shape[0])
        return res

    @staticmethod
    def cost_(y: np.ndarray, y_hat: np.ndarray) -> float:
        """Computes the half mean squared error of two non-empty numpy.ndarray,
            without any for loop. The two arrays must have the same dimensions.
        Args:
            y: has to be an numpy.ndarray, a vector.
            y_hat: has to be an numpy.ndarray, a vector.
        Returns:
            The half mean squared error of the two vectors as a float.
            None if y or y_hat are empty numpy.ndarray.
            None if y and y_hat does not share the same dimensions.
        Raises:
            This function should not raise any Exceptions.
        """
        if y.shape != y_hat.shape:
            return None
        j_elem = MyLinearRegression.cost_elem_(y, y_hat)
        return np.sum(j_elem)

    def plot(self, x: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> None:
        plt.plot(x, y, "o")
        plt.plot(x, y_hat)

        plt.legend(['Dataset','Hypothesis'])
        plt.xlabel("Mileage (km)")
        plt.ylabel("Price of car")
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

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
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
        if x.shape != y.shape or self.theta.shape != (2, 1):
            return None
        norm_x = self.minmax(x)
        for _ in range(self.max_iter):
            gradient = self.gradient(norm_x, y)
            self.theta -= gradient
        utils.save_theta(self.theta)
        y_hat = utils.predict(norm_x, self.theta)
        self.plot(x, y, y_hat)

def main():
    mylr = MyLinearRegression(0.1)
    km, price = utils.get_data()
    mylr.fit(km, price)

if __name__ == "__main__":
    main()

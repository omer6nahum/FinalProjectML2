import pickle
import numpy as np
from scipy.linalg import pinv2
from numpy.linalg import LinAlgError, inv


class LinReg:
    def __init__(self):
        self.beta = None
        self.is_fitted = False

    def fit(self, X, y):
        """
        Fit the linear model by solving least squares optimization problem.
        :param X: explaining variables matrix (without ones column).
        :param y: explained variable vector.
        :return: coefficient vector estimator.
        """
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]

        X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        tmp = np.matmul(X.T, X)
        try:
            tmp = inv(tmp)
        except LinAlgError:
            tmp = pinv2(tmp)
        tmp = np.matmul(tmp, X.T)
        self.beta = np.matmul(tmp, y)
        self.is_fitted = True
        return self.beta

    def predict(self, X_new):
        """
        Predict explained variable values for new observations.
        :param X_new: explaining variables matrix (without ones column).
        :return: predicted explained variable vector.
        """
        X_new = np.append(X_new, np.ones((X_new.shape[0], 1)), axis=1)
        return np.matmul(X_new, self.beta)

    def save_params(self, path):
        """
        Save model params as pickle.
        :param path: path for the params pickle.
        :return: None
        """
        with open(path, 'wb') as f:
            pickle.dump(self.beta, f)

    def load_params(self, path):
        """
        Load model params as pickle.
        :param path: path of the loaded params.
        :return:
        """
        with open(path, 'rb') as f:
            self.beta = pickle.load(f)
        self.is_fitted = True

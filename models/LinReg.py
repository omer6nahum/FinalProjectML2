import pickle
import numpy as np
import pandas as pd
from scipy.linalg import pinv2
from numpy.linalg import LinAlgError, inv
from Preprocess import create_train_test


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

    def pickle_params(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.beta, f)

    def load_params(self, path):
        with open(path, 'rb') as f:
            self.beta = pickle.load(f)
        self.is_fitted = True


if __name__ == '__main__':
    # LinReg, First Approach:
    x_train, x_test, y_train, y_test, z_train, z_test = create_train_test(test_year=21,
                                                                          approach=1,
                                                                          prefix_path='../')

    # Fit and predict:
    lin_reg = LinReg()
    lin_reg.fit(x_train, y_train)
    y_pred = lin_reg.predict(x_test)

    # Create tables
    # TODO(omermadmon): this should be a function of FirstApproach class):
    clubs_2021_pred = sorted(zip(z_test, y_pred), key=lambda x: x[1], reverse=True)
    clubs_2021_true = sorted(zip(z_test, y_test), key=lambda x: x[1], reverse=True)  # TODO: load data of real table
    predicted_table = pd.DataFrame(data=clubs_2021_pred)
    true_table = pd.DataFrame(data=clubs_2021_true)
    print(predicted_table)
    print(true_table)

    # Pickle and load model params:
    lin_reg.pickle_params('../pickles/models/lin_reg_ver1.pkl')
    lin_reg.load_params('../pickles/models/lin_reg_ver1.pkl')

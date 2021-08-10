import pickle

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import pinv2
from numpy.linalg import LinAlgError, inv
from Preprocess import create_train_test


class LinReg:
    def __init__(self):
        self.X = None
        self.y = None
        self.beta = None
        self.n = None
        self.k = None
        self.p = None
        self.anova_table = None

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
        # assert y.shape[1] == 1

        self.X, self.y = np.append(X, np.ones((X.shape[0], 1)), axis=1), y
        self.n, self.k, self.p = self.X.shape[0], self.X.shape[1] - 1, self.X.shape[1]
        tmp = np.matmul(self.X.T, self.X)
        try:
            tmp = inv(tmp)
        except LinAlgError:
            tmp = pinv2(tmp)
        tmp = np.matmul(tmp, self.X.T)
        self.beta = np.matmul(tmp, self.y)
        return self.beta

    def predict(self, X_new):
        """
        Predict explained variable values for new observations.
        :param X_new: explaining variables matrix (without ones column).
        :return: predicted explained variable vector.
        """
        X_new = np.append(X_new, np.ones((X_new.shape[0], 1)), axis=1)
        return np.matmul(X_new, self.beta)

    # TODO: find an alternative for ANOVA for the case k > n.
    def ANOVA(self):
        """
        Perform F test and calculate all SS, MS values.
        :return: ANOVA table.
        """
        SST = sum([(y_i - self.y.mean()) ** 2 for y_i in self.y])
        y_est = np.matmul(self.X, self.beta)
        SSRes = sum([(y_i - y_i_est) ** 2 for y_i, y_i_est in zip(self.y, y_est)])
        MSRes = SSRes / (self.n - self.p)
        SSR = SST - SSRes
        MSR = SSR / self.k
        MST = SST / (self.n - 1)
        F = MSR / MSRes
        pvalue = 1 - stats.f.cdf(F, self.k, self.n - self.p)

        Regression = {
            'SS': SSR,
            'df': self.k,
            'MS': MSR,
            'F-ratio': F,
            'P value': pvalue
        }

        Residuals = {
            'SS': SSRes,
            'df': self.n - self.p,
            'MS': MSRes,
            'F-ratio': '---',
            'P value': '---'
        }

        Total = {
            'SS': SST,
            'df': self.n - 1,
            'MS': MST,
            'F-ratio': '---',
            'P value': '---'
        }

        data = [Regression, Residuals, Total]
        index = ['Regression', 'Residuals', 'Total']
        self.anova_table = pd.DataFrame(data=data, index=index)
        return self.anova_table

    def get_R_square(self):
        """
        :return: R^2
        """
        return self.anova_table.loc['Regression']['SS'] / self.anova_table.loc['Total']['SS']

    def get_adjusted_R_square(self):
        """
        :return: adjusted R^2
        """
        return 1 - (self.anova_table.loc['Residuals']['MS'] / self.anova_table.loc['Total']['MS'])

    def pickle_params(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.beta, f)

    def load_params(self, path):
        with open(path, 'rb') as f:
            self.beta = pickle.load(f)


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
    lin_reg.pickle_params('../data/pickles/models/lin_reg_ver1.pkl')
    lin_reg.load_params('../data/pickles/models/lin_reg_ver1.pkl')

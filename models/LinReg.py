import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import pinv2
from numpy.linalg import LinAlgError, inv


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
        self.n, self.k, self.p = X.shape[0], X.shape[1] - 1, X.shape[1]
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

    def ANOVA(self):
        """
        Perform F test and calculate all SS, MS values.
        :return: ANOVA table.
        """
        SST = sum([(y_i - self.y.mean())**2 for y_i in self.y])
        y_est = np.matmul(self.X, self.beta)
        SSRes = sum([(y_i - y_i_est)**2 for y_i, y_i_est in zip(self.y, y_est)])
        MSRes = SSRes / (self.n-self.p)
        SSR = SST - SSRes
        MSR = SSR / self.k
        MST = SST/(self.n-1)
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
        pass

    def load_params(self, path):
        pass
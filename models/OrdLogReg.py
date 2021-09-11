import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from deps import LABELS


class OrdLogReg:
    def __init__(self):
        self.model1 = None  # H or not H  == > 0
        self.model2 = None  # not A or A  == > 1
        self.labels = LABELS
        self.is_fitted = False

    def fit(self, X, y):
        """
        Fit the linear models by solving Log Likelihood optimization problem for each model.
        :param X: explaining variables matrix (without ones column).
        :param y: explained variable vector (categorical label).
        :return: coefficient vectors estimators and intercepts estimators for each model.
        """
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]

        y = np.array([self.labels[y_i] for y_i in y])
        y1 = y > 0
        y2 = y > 1
        logreg_model1 = LogisticRegression(multi_class='ovr', max_iter=100, solver='lbfgs')  # binary classifier
        logreg_model1.fit(X, y1)
        logreg_model2 = LogisticRegression(multi_class='ovr', max_iter=100, solver='lbfgs')  # binary classifier
        logreg_model2.fit(X, y2)

        beta1 = logreg_model1.coef_
        intercept1 = logreg_model1.intercept_
        beta2 = logreg_model1.coef_
        intercept2 = logreg_model1.intercept_

        self.model1 = logreg_model1
        self.model2 = logreg_model2
        self.is_fitted = True

        return beta1, intercept1, beta2, intercept2

    def predict(self, X_new):
        """
        Predict probability distribution for the explained variable classes for new observations.
        :param X_new: explaining variables matrix (without ones column).
        :return: predicted probability distribution over the explained variable labels (for each new point).
        """
        pred1 = self.model1.predict_proba(X_new)  # ['H', 'D'+'A']
        pred2 = self.model2.predict_proba(X_new)  # ['H'+'D', 'A']
        res = np.zeros((pred1.shape[0], 3))
        res[:, 0] = pred1[:, 0]  # H
        res[:, 2] = pred2[:, 1]  # A
        res[:, 1] = 1 - res[:, 0] - res[:, 2]  # D
        return res

    def save_params(self, path):
        """
        Save models params as pickle.
        :param path: path for the params pickle.
        :return: None
        """
        with open(path, 'wb') as f:
            pickle.dump((self.model1, self.model2), f)

    def load_params(self, path):
        """
        Load models params as pickle.
        :param path: path of the loaded params.
        :return:
        """
        with open(path, 'rb') as f:
            self.model1, self.model2 = pickle.load(f)
        self.is_fitted = True

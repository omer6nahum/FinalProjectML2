import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from deps import LABELS


class LogReg:
    def __init__(self):
        self.model = None
        self.labels = LABELS
        self.is_fitted = False

    def fit(self, X, y):
        """
        Fit the linear model by solving Log Likelihood optimization problem.
        :param X: explaining variables matrix (without ones column).
        :param y: explained variable vector (categorical label).
        :return: coefficient vector estimator and intercept estimator.
        """
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]

        y = np.array([self.labels[y_i] for y_i in y])
        logreg_model = LogisticRegression(multi_class='multinomial', max_iter=100, solver='lbfgs')
        logreg_model.fit(X, y)

        beta = logreg_model.coef_
        intercept = logreg_model.intercept_
        self.model = logreg_model
        self.is_fitted = True

        return beta, intercept

    def predict(self, X_new):
        """
        Predict probability distribution for the explained variable classes for new observations.
        :param X_new: explaining variables matrix (without ones column).
        :return: predicted probability distribution over the explained variable labels (for each new point).
        """
        return self.model.predict_proba(X_new)

    def save_params(self, path):
        """
        Save model params as pickle.
        :param path: path for the params pickle.
        :return: None
        """
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_params(self, path):
        """
        Load model params as pickle.
        :param path: path of the loaded params.
        :return:
        """
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True

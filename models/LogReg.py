import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from Preprocess import create_train_test


class LogReg:
    def __init__(self):
        # self.beta = None
        # self.intercept = None
        self.model = None

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

        logreg_model = LogisticRegression(multi_class='multinomial')
        logreg_model.fit(X, y)

        beta = logreg_model.coef_
        intercept = logreg_model.intercept_
        self.model = logreg_model

        return beta, intercept

    def predict(self, X_new):
        """
        Predict probability distribution for the explained variable classes for new observations.
        :param X_new: explaining variables matrix (without ones column).
        :return: predicted probability distribution over the explained variable labels (for each new point).
        """
        return self.model.predict_proba(X_new)

    def pickle_params(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_params(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


if __name__ == '__main__':
    # LogReg, First Approach:
    x_train, x_test, y_train, y_test, z_train, z_test = create_train_test(test_year=21, approach=2, prefix_path='../')
    print(0)
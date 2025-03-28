import numpy as np
from utils import sigmoid

class LogReg:
    """
    Logistic Regression using Mean Squared Error as Cost Function.

    Parameters
    ----------
    learning rate : default = 0.001
        step size at each iteration while moving toward a minimum of a loss function.

    max_iter : default = 5000
        number of iterations.
    """
    def __init__(self, learning_rate = 0.001, max_iter = 5000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.betas = None
        
    def fit(self, X, y):
        """
        Fit function to fit the model.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data.
        
        y : array of shape (n_samples, 1)
            Target data.
        """
        n_samples, n_features = X.shape
        X = np.append(np.ones((X.shape[0],1)), X, axis=1)
        y = y.reshape(n_samples,1)
        self.betas = np.zeros((n_features + 1, 1))

        for i in range(self.max_iter):
            y_linear = np.dot(X, self.betas)
            y_pred = sigmoid(y_linear)
            error = y - y_pred
            d_betas = (-2/n_samples)*np.dot(X.T, error)
            self.betas = self.betas - self.learning_rate*d_betas

    def predict(self, X):
        """
        Predict function.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        y_pred : array of shape (n_samples, 1)
            Predicted array of binary class.
        """
        n_samples = X.shape[0]
        X = np.append(np.ones((X.shape[0],1)), X, axis=1)
        y_pred_linear = np.dot(X, self.betas).reshape(n_samples,)
        y_pred = np.array([1 if y > 0.5 else 0 for y in y_pred_linear])
        return y_pred
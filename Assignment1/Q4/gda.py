# Imports - you can add any other permitted libraries
import numpy as np
from scipy.stats import multivariate_normal
# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.


class GaussianDiscriminantAnalysis:
    # Assume Binary Classification
    def __init__(self):
        self.num_features : int = None
        self.num_data_points : int = None
        self.X : np.array = None
        self.y : np.array = None 
        self.mean : float = None
        self.std : float = None
        
        self.phi : float = None
        self.mu_0 : np.array = None
        self.mu_1 : np.array = None
        self.sigma : np.array = None
        self.sigma_0 : np.array = None
        self.sigma_1 : np.array = None
        self.assume_same_covariance : bool = False
        pass
    
    def fit(self, X, y, assume_same_covariance=False):
        """
        Fit the Gaussian Discriminant Analysis model to the data.
        Remember to normalize the input data X before fitting the model.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data.
            
        y : numpy array of shape (n_samples,)
            The target labels - 0 or 1.
        
        learning_rate : float
            The learning rate to use in the update rule.
        
        Returns
        -------
        Parameters: 
            If assume_same_covariance = True - 3-tuple of numpy arrays mu_0, mu_1, sigma 
            If assume_same_covariance = False - 4-tuple of numpy arrays mu_0, mu_1, sigma_0, sigma_1
            The parameters learned by the model.
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X = (X - self.mean) / (self.std + 1e-8)
        self.X = X

        self.y = y
        self.num_data_points, self.num_features = self.X.shape
        self.assume_same_covariance = assume_same_covariance
        
        self.phi = np.mean(self.y)
        self.mu_0 = np.mean(self.X[y == 0], axis=0)
        self.mu_1 = np.mean(self.X[y == 1], axis=0)
        n = self.num_data_points

        if self.assume_same_covariance:
            self.sigma = np.zeros((self.num_features, self.num_features))
            for i in range(n):
                x_i = self.X[i] - (self.mu_1 if self.y[i] == 1 else self.mu_0)
                self.sigma += np.outer(x_i, x_i)
            self.sigma /= n
            return self.mu_0, self.mu_1, self.sigma
        
        else:
            self.sigma_0 = np.zeros((self.num_features, self.num_features))
            self.sigma_1 = np.zeros((self.num_features, self.num_features))
            
            count_0 = np.sum(y == 0)
            count_1 = np.sum(y == 1)
            for i in range(n):
                if self.y[i] == 0:
                    x_i = self.X[i] - self.mu_0
                    self.sigma_0 += np.outer(x_i, x_i)
                else:
                    x_i = self.X[i] - self.mu_1
                    self.sigma_1 += np.outer(x_i, x_i)

            self.sigma_0 /= count_0
            self.sigma_1 /= count_1

            return self.mu_0, self.mu_1, self.sigma_0, self.sigma_1

    def predict_proba(self, X):
        """
        Compute the probability of belonging to class 1 using Bayes' rule.
        """
        X = (X - self.mean) / (self.std + 1e-8)
        # X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        if self.assume_same_covariance:
            p_x_given_y0 = multivariate_normal.pdf(X, mean=self.mu_0, cov=self.sigma, allow_singular=True)
            p_x_given_y1 = multivariate_normal.pdf(X, mean=self.mu_1, cov=self.sigma, allow_singular=True)
        else:
            p_x_given_y0 = multivariate_normal.pdf(X, mean=self.mu_0, cov=self.sigma_0, allow_singular=True)
            p_x_given_y1 = multivariate_normal.pdf(X, mean=self.mu_1, cov=self.sigma_1, allow_singular=True)
        
        p_y1_x = (self.phi * p_x_given_y1) / ((self.phi * p_x_given_y1) + ((1 - self.phi) * p_x_given_y0))
        return p_y1_x

    def predict(self, X):
        """
        Predict the target values for the input data.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data.
            
        Returns
        -------
        y_pred : numpy array of shape (n_samples,)
            The predicted target label.
        """
        return (self.predict_proba(X) >= 0.5).astype(int)
# Imports - you can add any other permitted libraries
import numpy as np

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
        self.theta : np.array = None
        self.num_iter : int = int(10)
        self.eps : float = 0
        self.mean : float = None
        self.std : float = None
        
        self.phi = None
        self.mu_0 = None
        self.mu_1 = None
        self.sigma = None
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
        self.X = np.hstack((np.ones((X.shape[0], 1)), X)) # Add a column of ones to X for the intercept term
        self.y = y
        self.num_data_points, self.num_features = self.X.shape
        self.theta = np.zeros(self.num_features)
        n_iter = self.num_iter
        prev_loss : float = float('inf')
        
        self.phi = np.mean(y)
        self.mu_0 = np.mean(X[y == 0], axis=0)
        self.mu_1 = np.mean(X[y == 1], axis=0)
        n = self.X.shape[0]

        for i in range(n_iter):
            x_i = X[i] - (self.mu_1 if y[i] == 1 else self.mu_0)
            self.sigma += np.outer(x_i, x_i)

        self.sigma /= n
        
        return self.mu_0, self.mu_1, self.sigma
    
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
        def predict_proba(self, X):
            """
            Compute the probability of belonging to class 1 using Bayes' rule.
            """
            from scipy.stats import multivariate_normal
            p_x_given_y0 = multivariate_normal.pdf(X, mean=self.mu_0, cov=self.sigma)
            p_x_given_y1 = multivariate_normal.pdf(X, mean=self.mu_1, cov=self.sigma)
            p_y1_x = (self.phi * p_x_given_y1) / ((self.phi * p_x_given_y1) + ((1 - self.phi) * p_x_given_y0))
            return p_y1_x
        return (self.predict_proba(X) >= 0.5).astype(int)
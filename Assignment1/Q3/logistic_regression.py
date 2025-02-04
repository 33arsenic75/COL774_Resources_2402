# Imports - you can add any other permitted libraries
import numpy as np

# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

class LogisticRegressor:
    # Assume Binary Classification
    def __init__(self):
        self.num_features : int = None
        self.num_data_points : int = None
        self.X : np.array = None
        self.y : np.array = None 
        self.theta : np.array = None
        self.num_iter : int = int(10)
        self.eps : float = 0
        self.loss_data : list = []
        self.mean : float = None
        self.std : float = None
        pass
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y, learning_rate=0.01):
        """
        Fit the linear regression model to the data using Newton's Method.
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
        List of Parameters: numpy array of shape (n_iter, n_features+1,)
            The list of parameters obtained after each iteration of Newton's Method.
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

        for i in range(n_iter):
            z = np.dot(self.X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(self.X.T, (h - self.y))
            R = np.diag(h * (1 - h))
            H = np.dot(self.X.T, np.dot(R, self.X))
            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                print("Hessian is singular, stopping updates.")
                break
            
            delta_theta = np.dot(H_inv, gradient)
            self.theta -= delta_theta
            # Compute loss (negative log-likelihood)
            loss = -np.sum(y * np.log(h + 1e-8) + (1 - y) * np.log(1 - h + 1e-8))
            self.loss_data.append((self.theta.copy(), loss))

            
            # Check for convergence
            if np.linalg.norm(delta_theta) < self.eps:
                break

        return self.theta

        
    
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
        def predict_proba(X):
            X = (X - self.mean) / (self.std + 1e-8)
            X = np.hstack((np.ones((X.shape[0], 1)), X))
            return self.sigmoid(np.dot(X, self.theta))
        
        return (predict_proba(X) >= 0.5).astype(int)

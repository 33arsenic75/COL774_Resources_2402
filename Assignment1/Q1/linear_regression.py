# Imports - you can add any other permitted libraries
import numpy as np

# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

class LinearRegressor:
    def __init__(self):
        self.num_features : int = None
        self.num_data_points : int = None
        self.X : np.array = None
        self.y : np.array = None 
        self.theta : np.array = None
        self.num_iter : int = 1000
        pass
    
    def check_convergence(self, theta, learning_rate):
        """
        Check if the gradient descent has converged by checking if the change in the value of the loss function is less than 1e-4.
        
        Parameters
        ----------
        theta : numpy array of shape (n_features,)
            The current parameter values.
            
        learning_rate : float
            The learning rate to use in the update rule.
            
        Returns
        -------
        bool
            True if the change in the value of the loss function is less than 1e-4, False otherwise.
        """
        pass

    def fit(self, X, y, learning_rate=0.01):
        """
        Fit the linear regression model to the data using Gradient Descent.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data.
            
        y : numpy array of shape (n_samples,)
            The target values.

        learning_rate : float
            The learning rate to use in the update rule.
            
        Returns
        -------
        List of Parameters: numpy array of shape (n_iter, n_features,)
            The list of parameters obtained after each iteration of Gradient Descent.
        """


        self.X = np.hstack((np.ones((X.shape[0], 1)), X)) # Add a column of ones to X for the intercept term
        self.y = y
        self.num_data_points, self.num_features = self.X.shape
        self.theta = np.zeros(self.num_features)
        print("X: ", self.X.shape)
        print("y: ", self.y.shape)
        print("Num data points: ", self.num_data_points)
        print("Num features: ", self.num_features)
        n_iter = self.num_iter
        # Gradient Descent
        for iter in range(n_iter):
            predictions = np.dot(self.X, self.theta)  # Calculate predictions
            errors = predictions - self.y              # Calculate errors
            loss = (1 / (2 * self.num_data_points)) * np.sum(errors ** 2)
            print(f"Iteration {iter+1} - Loss: {loss}")
            gradient = (1 / self.num_data_points) * np.dot(self.X.T, errors)  # Compute gradient
            
            # Ensure gradient is 1D and matches the shape of theta
            self.theta -= learning_rate * gradient.flatten()  # Update theta
        
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
            The predicted target values.
        """
        pass
# Imports - you can add any other permitted libraries
import numpy as np

# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

import numpy as np

def generate(N, theta, input_mean, input_sigma, noise_sigma):
    """
    Generate normally distributed input data and target values.
    
    Note that we have 2 input features.
    
    Parameters
    ----------
    N : int
        The number of samples to generate.
        
    theta : numpy array of shape (3,)
        The true parameters of the linear regression model.
        
    input_mean : numpy array of shape (2,)
        The mean of the input data.
        
    input_sigma : numpy array of shape (2,)
        The standard deviation of the input data.
        
    noise_sigma : float
        The standard deviation of the Gaussian noise.
        
    Returns
    -------
    X : numpy array of shape (N, 2)
        The input data.
        
    y : numpy array of shape (N,)
        The target values.
    """
    
    # Generate input data X from a normal distribution
    X = np.random.normal(loc=input_mean, scale=input_sigma, size=(N, 2))
    
    # Calculate the target values y using the linear model y = theta0 + theta1*x1 + theta2*x2 + noise
    noise = np.random.normal(loc=0, scale=noise_sigma, size=N)
    y = theta[0] + theta[1] * X[:, 0] + theta[2] * X[:, 1] + noise
    
    return X, y

class StochasticLinearRegressor:
    def __init__(self):
        self.num_features : int = None
        self.num_data_points : int = None
        self.X : np.array = None
        self.y : np.array = None 
        self.theta : np.array = None
        self.num_iter : int = int(1e4)
        self.eps : float = 1e-7
        self.loss_data : list = []
        self.batch_size : int = None
        self.theta_history : list = []

    def closed_form_solution(self, X, y):
        """
        Compute the closed form solution for linear regression.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data.
            
        y : numpy array of shape (n_samples,)
            The target values.
            
        Returns
        -------
        theta : numpy array of shape (n_features,)
            The parameters of the linear regression model.
        """
        self.X = np.hstack((np.ones((X.shape[0], 1)), X)) # Add a column of ones to X for the intercept term
        self.y = y
        theta_closed_form = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(y)
        return theta_closed_form

    def fit(self, X, y, learning_rate=0.01, batch_size=1000):
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
        n_iter = self.num_iter
        self.batch_size = batch_size
        
        prev_loss : float = float('inf')
        for iter in range(n_iter):
            indices = np.arange(self.num_data_points)
            np.random.shuffle(indices)
            X_shuffled = self.X[indices]
            y_shuffled = self.y[indices]
            total_loss = 0
            for i in range(0, self.num_data_points, self.batch_size):
                # Create mini-batch
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                predictions = np.dot(X_batch, self.theta)
                errors = predictions - y_batch
                loss = np.sum(errors ** 2)
                total_loss += loss
                gradient = (1 / len(y_batch)) * np.dot(X_batch.T, errors)
                self.theta -= learning_rate * gradient.flatten()  # Ensure gradient is 1D

            total_loss = total_loss / (2 * self.num_data_points)
            if abs(prev_loss - total_loss) < self.eps:
                break
            prev_loss = total_loss
            self.loss_data.append([self.theta.copy(), total_loss])
            # if(iter%500 == 499):
            #     print(f"Epoch {iter + 1} - Loss: {total_loss}")
            self.theta_history.append(self.theta.copy())

        return np.array(self.theta_history)
    
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
        X_pred = np.hstack((np.ones((X.shape[0], 1)), X)) # Add a column of ones to X for the intercept term
        y_pred = np.dot(X_pred, self.theta)  # Calculate predictions
        return y_pred
    
    def loss(self,X,y):
        """
        Return the loss value after the last iteration.
        
        Returns
        -------
        loss : float
            The loss value after the last iteration.
        """
        y_pred = self.predict(X)
        errors = y_pred - y
        losss = (1 / (2 * X.shape[0])) * np.sum(errors ** 2)
        return losss
from gda import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

def plot_theta(theta):
    theta0 = np.array([item[0][0] for item in theta])
    theta1 = np.array([item[0][1] for item in theta])
    theta2 = np.array([item[0][2] for item in theta])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the data
    ax.scatter(theta0, theta1, c=theta2, cmap='viridis')

    # Adding labels
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')
    ax.set_zlabel('Theta 2')

    # Show the plot
    plt.savefig(f'q2_4.png')
    
def q4_1(X,y):
    print("Q4.1")
    model = GaussianDiscriminantAnalysis()
    mu_0, mu_1, sigma = model.fit(X, y, assume_same_covariance=True)
    print(f"mu_0: {mu_0}")
    print(f"mu_1: {mu_1}")
    print(f"sigma: {sigma}")
    print("Normalization Constant")
    print(f"Mean: {model.mean}")
    print(f"Std: {model.std}")
    print("--"*20)
    
def q4_2_3(X,y):
    print("Q4.2")
    model = GaussianDiscriminantAnalysis()
    mu_0, mu_1, sigma = model.fit(X, y, assume_same_covariance=True)
    # Plot data points
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Class 0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class 1')

    # Create a mesh grid for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    sigma_inv = np.linalg.inv(sigma)
    w = sigma_inv @ (mu_1 - mu_0)
    b = -0.5 * (mu_1.T @ sigma_inv @ mu_1 - mu_0.T @ sigma_inv @ mu_0) + np.log(np.sum(y) / (len(y) - np.sum(y)))
    # print(w)
    # print(b)
    # Predict on the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='dashed', linewidths=2)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.title('Logistic Regression Decision Boundary')
    plt.savefig('q4_2.png')
    print(f"Image saved as q4_2.png")
    plt.close()
    print("--"*20)

def q4_4(X,y):
    print("Q4.4")
    model = GaussianDiscriminantAnalysis()
    mu_0, mu_1, sigma_0, sigma_1 = model.fit(X, y, assume_same_covariance=False)
    print(f"mu_0: {mu_0}")
    print(f"mu_1: {mu_1}")
    print(f"sigma_0: {sigma_0}")
    print(f"sigma_1: {sigma_1}")
    print("--"*20)

def q4_5(X,y):
    print("Q4.5")
    model = GaussianDiscriminantAnalysis()
    mu_0, mu_1, sigma_0, sigma_1 = model.fit(X, y, assume_same_covariance=False)
    phi = model.phi
    inv_sigma_0 = np.linalg.inv(sigma_0)
    inv_sigma_1 = np.linalg.inv(sigma_1)
    A = inv_sigma_1 - inv_sigma_0
    B = -2 * (mu_1.T @ inv_sigma_1 - mu_0.T @ inv_sigma_0)
    C = mu_1.T @ inv_sigma_1 @ mu_1 - mu_0.T @ inv_sigma_0 @ mu_0
    C -= 2 * np.log(phi / (1 - phi)) + 2 * np.log(np.linalg.det(sigma_0) / np.linalg.det(sigma_1))
    print(f"A: {A}")
    print(f"B: {B}")
    print(f"C: {C}")

    # Plot data points
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Class 0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class 1')

    # Create a mesh grid for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Predict on the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='dashed', linewidths=2)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.title('Logistic Regression Decision Boundary')
    plt.savefig('q4_5.png')
    plt.close()
    print(f"Image saved as q4_5.png")
    print("--"*20)


X = np.loadtxt('../data/Q4/q4x.dat', delimiter=None)
y = np.genfromtxt('../data/Q4/q4y.dat', dtype=str)

mp = {'Alaska': 0, 'Canada': 1}
y = np.array([mp[i] for i in y])

q4_1(X,y)
# q4_2_3(X,y)
# q4_4(X,y)
q4_5(X,y)

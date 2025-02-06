from gda import *
import pandas as pd
import matplotlib.pyplot as plt

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
    model = GaussianDiscriminantAnalysis()
    mu_0, mu_1, sigma = model.fit(X, y, assume_same_covariance=True)
    print(f"mu_0: {mu_0}")
    print(f"mu_1: {mu_1}")
    print(f"sigma: {sigma}")
    
def q4_2(X,y):
    model = GaussianDiscriminantAnalysis()
    mu_0, mu_1, sigma = model.fit(X, y, assume_same_covariance=True)
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

    pos = np.dstack((xx, yy))
    epsilon = 1e-6
    rv_0 = multivariate_normal(mean=model.mu_0[1:], cov=model.sigma[1:, 1:] + np.eye(2) * epsilon)
    rv_1 = multivariate_normal(mean=model.mu_1[1:], cov=model.sigma[1:, 1:] + np.eye(2) * epsilon)


    grid = np.c_[xx.ravel(), yy.ravel()]
    pdf_0 = rv_0.pdf(grid).reshape(xx.shape)
    pdf_1 = rv_1.pdf(grid).reshape(xx.shape)

    plt.contour(xx, yy, pdf_0, levels=np.logspace(-5, 0, 10), cmap="Reds", alpha=0.5)
    plt.contour(xx, yy, pdf_1, levels=np.logspace(-5, 0, 10), cmap="Blues", alpha=0.5)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.title('Logistic Regression Decision Boundary')
    plt.savefig('q4_2.png')
    print(f"Image saved as q4_2.png")

def q4_4(X,y):
    model = GaussianDiscriminantAnalysis()
    mu_0, mu_1, sigma_0,sigma_1 = model.fit(X, y, assume_same_covariance=False)
    print(f"mu_0: {mu_0}")
    print(f"mu_1: {mu_1}")
    print(f"sigma_0: {sigma_0}")
    print(f"sigma_1: {sigma_1}")

X = np.loadtxt('../data/Q4/q4x.dat', delimiter=None)
y = np.genfromtxt('../data/Q4/q4y.dat', dtype=str)

mp = {'Alaska': 0, 'Canada': 1}
y = np.array([mp[i] for i in y])
# q4_1(X,y)
# q4_2(X,y)
q4_4(X,y)

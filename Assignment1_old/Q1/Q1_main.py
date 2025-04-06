from linear_regression import LinearRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def q1_1():
    print("Q1.1")
    X = np.array(pd.read_csv('../data/Q1/linearX.csv', header=None).values)
    y = np.array(pd.read_csv('../data/Q1/linearY.csv', header=None).values).flatten()  # Flatten y to 1D array
    for learning_rate in np.arange(0.001, 0.05, 0.001):
        model = LinearRegressor()
        theta_history = model.fit(X, y,learning_rate=learning_rate)
        theta = theta_history[-1]
        y_predict = model.predict(X=X)
        errors = y_predict - y
        loss = (1 / (2 * X.shape[0])) * np.sum(errors ** 2)
        print(f"eta: {learning_rate:.3f}, theta: {theta}, loss: {loss:3f}")
    print("--"*20)

def q1_2(learning_rate=0.018):
    print("Q1.2")
    X = np.array(pd.read_csv('../data/Q1/linearX.csv', header=None).values)
    y = np.array(pd.read_csv('../data/Q1/linearY.csv', header=None).values).flatten()  # Flatten y to 1D array
    model = LinearRegressor()
    theta_history = model.fit(X, y,learning_rate=learning_rate)
    theta = theta_history[-1]
    print(theta)
    y_pred = model.predict(X)
    errors = y_pred - y
    loss = (1 / (2 * X.shape[0])) * np.sum(errors ** 2)
    # print(f"eta: {learning_rate} , l: {loss}")
    y_line = theta[0] + theta[1] * X.flatten()  # Ensure X is flattened for calculation
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Data Points')  # Plot original data points
    plt.plot(X, y_line, color='red', label='Hypothesis (Linear Equation)')  # Plot linear equation line
    plt.title('Linear Regression Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'q1_2_{learning_rate}.png')
    plt.close()
    print(f"Image saved to q1_2_{learning_rate}")
    print("--"*20)



def q1_3(learning_rate=0.018):
    print("Q1.3")
    X = np.array(pd.read_csv('../data/Q1/linearX.csv', header=None).values)
    y = np.array(pd.read_csv('../data/Q1/linearY.csv', header=None).values).flatten()  # Flatten y to 1D array
    model = LinearRegressor()
    theta_history = model.fit(X, y,learning_rate=learning_rate)
    theta = theta_history[-1]
    loss_data = model.loss_data
    theta0 = np.array([item[0][0] for item in loss_data])
    theta1 = np.array([item[0][1] for item in loss_data])
    loss = np.array([item[1] for item in loss_data])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the data
    ax.scatter(theta0, theta1, loss, c=loss, cmap='viridis')

    # Adding labels
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')
    ax.set_zlabel('Loss')

    # Show the plot
    plt.savefig(f'q1_3_{learning_rate}.png')
    plt.close()
    print("--"*20)

    
def q1_4(learning_rate=0.018):
    print("Q1.4")
    X = np.array(pd.read_csv('../data/Q1/linearX.csv', header=None).values)
    y = np.array(pd.read_csv('../data/Q1/linearY.csv', header=None).values).flatten()  # Flatten y to 1D array
    model = LinearRegressor()
    theta_history = model.fit(X, y,learning_rate=learning_rate)
    theta = theta_history[-1]
    loss_data = model.loss_data
    theta0 = np.array([item[0][0] for item in loss_data])
    theta1 = np.array([item[0][1] for item in loss_data])
    loss = np.array([item[1] for item in loss_data])
    fig, ax = plt.subplots()
    scatter = ax.scatter(theta0, theta1, c=loss, cmap='viridis', edgecolors='k')

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Loss')

    # Labels
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')
    ax.set_title('Loss Heatmap')
    ax.grid(True)
    # Save and show
    plt.savefig(f'q1_4_{learning_rate}.png')
    plt.close()
    print("--"*20)

def q1_5():
    print("Q1.5")
    X = np.array(pd.read_csv('../data/Q1/linearX.csv', header=None).values)
    y = np.array(pd.read_csv('../data/Q1/linearY.csv', header=None).values).flatten()  # Flatten y to 1D array
    # fig, ax = plt.subplots()
    for learning_rate in [0.001,0.025,0.1]:
        fig, ax = plt.subplots()
        model = LinearRegressor()
        theta_history = model.fit(X, y,learning_rate=learning_rate)
        theta = theta_history[-1]
        loss_data = model.loss_data
        theta0 = np.array([item[0][0] for item in loss_data])
        theta1 = np.array([item[0][1] for item in loss_data])
        loss = np.array([item[1] for item in loss_data])
        
        scatter = ax.scatter(theta0, theta1, c=loss, cmap='viridis', edgecolors='k')

        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Loss')

        # Labels
        ax.set_xlabel('Theta 0')
        ax.set_ylabel('Theta 1')
        ax.set_title('Loss Heatmap')
        ax.grid(True)
        # Save and show
        plt.savefig(f'q1_5_{learning_rate}.png')
        plt.close()
        print("--"*5)
    print("--"*20)


q1_1()
q1_2()
q1_3()
q1_4()
q1_5()
    
    

    

from sampling_sgd import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def get_data(file_name, split_ratio=0.8):
    df = pd.read_csv(file_name)
    X = df[['Feature_1', 'Feature_2']].to_numpy()
    y = df['Target'].to_numpy()

    data = np.column_stack((X, y))

    # Shuffle the data
    np.random.shuffle(data)

    # Calculate the split index for an 80-20 split
    split_index = int(split_ratio * len(data))

    # Split the data into training and testing sets
    train_data = data[:split_index]
    test_data = data[split_index:]

    # Separate the features and target for training and testing sets
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    return X_train, y_train, X_test, y_test

def plot_theta(loss_data, batch_size):
    theta0 = np.array([item[0][0] for item in loss_data])
    theta1 = np.array([item[0][1] for item in loss_data])
    theta2 = np.array([item[0][2] for item in loss_data])
    loss = np.array([item[1] for item in loss_data])

    # Create a figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Ensure 3D projection

    # 3D scatter plot
    scatter = ax.scatter(theta0, theta1, theta2, c=loss, cmap='viridis', edgecolors='k')

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Loss')
    ax.plot(theta0, theta1, theta2, color='red', linestyle='-', marker='o', markersize=3, alpha=0.7)
    # Axis labels
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')

    ax.set_title('Loss Heatmap in 3D')

    # Save and show plot
    plt.savefig(f'q2_5_{batch_size}.png')
    plt.close()
    print(f"File saved as q2_5_{batch_size}.png")

def q2_5(X_train, y_train, X_test, y_test):
    print("Q2.5")
    # batch_size_list = [ 1, 80, 80_000, 800_000]
    model = StochasticLinearRegressor()
    model.fit(X_train, y_train,learning_rate=0.001)
    theta_sgd_history = model.theta_history
    test_error = model.loss(X=X_test,y=y_test)
    loss_data = model.loss_data
    theta_sgd = model.theta
    i = 0
    print("--"*5)
    for batch_size in model.batch_size_list:
        print(f"Batch Size: {batch_size}")
        print(f"Iterations: {len(theta_sgd_history[batch_size])}")
        print(f"Parameters: {theta_sgd[batch_size]}")
        print(f"Train MSE: {loss_data[batch_size][-1][-1]}")
        print(f"Test Loss: {test_error[i]}")
        i += 1
        plot_theta(loss_data[batch_size],batch_size)
        print("--"*5)
    print("--"*20)

def q2_3b(X_train, y_train, X_test, y_test):
    print("Q2.3b")
    start_time = time.time()
    model = StochasticLinearRegressor()
    theta_closed = model.closed_form_solution(X_train, y_train)
    end_time = time.time()
    print(f"Closed Form Solution: {theta_closed}")
    print(f"Time Taken: {end_time-start_time}")
    print("--"*20)




X,y = generate(N = 1000000, theta = np.array([3, 1, 2]), input_mean = np.array([3, -1]), input_sigma = np.array([4, 4]), noise_sigma = 2)
df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
df['Target'] = y
df.to_csv('generated_data.csv', index=False)

X_train, y_train, X_test, y_test = get_data('generated_data.csv')


q2_5(X_train,y_train,X_test,y_test)
# q2_3b(X_train,y_train,X_test,y_test)


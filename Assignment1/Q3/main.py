from logistic_regression import *
import pandas as pd
import matplotlib.pyplot as plt

def get_data(file_name):
    learning_rate = 0.01
    df = pd.read_csv(file_name)
    X = df[['Feature_1', 'Feature_2']].to_numpy()
    y = df['Target'].to_numpy()

    data = np.column_stack((X, y))

    # Shuffle the data
    np.random.shuffle(data)

    # Calculate the split index for an 80-20 split
    split_index = int(0.8 * len(data))

    # Split the data into training and testing sets
    train_data = data[:split_index]
    test_data = data[split_index:]

    # Separate the features and target for training and testing sets
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    return X_train, y_train, X_test, y_test

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
    



X = np.array(pd.read_csv('../data/Q3/logisticX.csv', header=None).values)
y = np.array(pd.read_csv('../data/Q3/logisticY.csv', header=None).values).flatten()  # Flatten y to 1D array


# loss_data = model.loss_data
# # theta_ivt = model.closed_form_solution(X_train, y_train)
# for i in loss_data:
#     print(i)

# print(theta_lgst)
model = LogisticRegressor()
theta_lgst = model.fit(X, y,learning_rate=0.01)
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
plt.title('Logistic Regression Decision Boundary')
plt.savefig('q3_1.png')
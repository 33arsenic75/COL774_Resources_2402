from gda import *
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
    



# X = np.array(pd.read_csv('../data/Q3/logisticX.csv', header=None).values)
# y = np.array(pd.read_csv('../data/Q3/logisticY.csv', header=None).values).flatten()  # Flatten y to 1D array

X = np.loadtxt('../data/Q4/q4x.dat', delimiter=None)
y = np.genfromtxt('../data/Q4/q4y.dat', dtype=str)

mp = {'Alaska': 0, 'Canada': 1}
y = np.array([mp[i] for i in y])

print(X.shape, y.shape)

model1 = GaussianDiscriminantAnalysis()
theta_gda1 = model1.fit(X, y, assume_same_covariance=True)
print([x.shape for x in theta_gda1])

model2 = GaussianDiscriminantAnalysis()
theta_gda2 = model2.fit(X, y, assume_same_covariance=False)
print([x.shape for x in theta_gda2])

# Plot data points
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Class 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class 1')

# Create a mesh grid for decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Predict on the grid
Z = model1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='dashed', linewidths=2)

pos = np.dstack((xx, yy))
epsilon = 1e-6
rv_0 = multivariate_normal(mean=model1.mu_0[1:], cov=model1.sigma[1:, 1:] + np.eye(2) * epsilon)
rv_1 = multivariate_normal(mean=model1.mu_1[1:], cov=model1.sigma[1:, 1:] + np.eye(2) * epsilon)


grid = np.c_[xx.ravel(), yy.ravel()]
pdf_0 = rv_0.pdf(grid).reshape(xx.shape)
pdf_1 = rv_1.pdf(grid).reshape(xx.shape)

plt.contour(xx, yy, pdf_0, levels=np.logspace(-5, 0, 10), cmap="Reds", alpha=0.5)
plt.contour(xx, yy, pdf_1, levels=np.logspace(-5, 0, 10), cmap="Blues", alpha=0.5)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Logistic Regression Decision Boundary')
plt.savefig('q4_1.png')
print("Done")
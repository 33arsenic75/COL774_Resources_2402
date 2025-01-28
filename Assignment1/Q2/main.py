from sampling_sgd import *
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
    


X,y = generate(N = 1000000, theta = np.array([3, 1, 2]), input_mean = np.array([3, -1]), input_sigma = np.array([4, 4]), noise_sigma = 2)
df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
df['Target'] = y
df.to_csv('generated_data.csv', index=False)

X_train, y_train, X_test, y_test = get_data('generated_data.csv')
model = StochasticLinearRegressor()
theta_sgd = model.fit(X_train, y_train,learning_rate=0.01, batch_size=8000)
# theta_ivt = model.closed_form_solution(X_train, y_train)
print(theta_sgd)
plot_theta(model.loss_data)

# print(theta_ivt)
# loss = model.loss(X=X_test,y=y_test)
# print(loss)

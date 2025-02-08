from logistic_regression import *
import pandas as pd
import matplotlib.pyplot as plt



def q3_1(X,y):
    print("Q3.1")
    model = LogisticRegressor()
    theta_lgst_history = model.fit(X, y,learning_rate=0.01)
    theta_lgst = theta_lgst_history[-1]
    print(f"Parameters: {theta_lgst}")
    print("Normalization Constants: ")
    print(f"Mean: {model.mean}")
    print(f"Std: {model.std}")
    print("--"*20)

def q3_2(X,y):
    print("Q3.2")
    model = LogisticRegressor()
    theta_lgst_history = model.fit(X, y,learning_rate=0.01)
    theta_lgst = theta_lgst_history[-1]   
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
    plt.savefig('q3_2.png') 
    print("Image saved as q3.png")
    plt.close()
    print("--"*20)


X = np.array(pd.read_csv('../data/Q3/logisticX.csv', header=None).values)
y = np.array(pd.read_csv('../data/Q3/logisticY.csv', header=None).values).flatten()

q3_1(X,y)
q3_2(X,y)
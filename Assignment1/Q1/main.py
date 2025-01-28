from linear_regression import LinearRegressor
import numpy as np
import pandas as pd


def main():
    # X = np.array(pd.read_csv('../data/Q1/linearX.csv', header=None).values)
    # y = np.array(pd.read_csv('../data/Q1/linearY.csv', header=None).values)
    X = np.array([[1],[2],[3],[4],[5]])
    y = np.array([2,4,6,8,10])
    model = LinearRegressor()
    theta = model.fit(X, y)
    print(theta)

main()
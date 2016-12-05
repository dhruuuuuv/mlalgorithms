import csv
import math
import operator

import matplotlib.pyplot as plt
import numpy as np

def linear_regression(filearg):
    # load data from file passed as arguement
    data = np.loadtxt(filearg, delimiter=' ')
    a = np.matrix(data)
    print(a)
    # split input into x and y
    x = a[:,0]
    y = a[:,1]

    ones = np.matrix([[1], [1], [1], [1]])
    xbar = np.ones((6, 2))
    # produce the x bar matrix
    xbar[:,:-1] = x
    # xbart = xbar.T

    result = np.linalg.inv((np.dot(xbar.T, xbar)))
    result = result.dot(xbar.T).dot(y)
    # print("linreg")
    w = result.item(0)
    b = result.item(1)
    return (w,b)


def mse(originals, predictions):
    n = len(originals)
    summed_val = 0
    for i in range(0, n):
        x = (predictions[i] - originals[i]) ** 2
        summed_val += x
    mse = summed_val / n
    return mse

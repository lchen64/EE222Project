import numpy as np
from nesterov import run_nesterov
import matplotlib.pyplot as plt

A = np.random.randn(100, 500)
b = np.random.randn(100, 1) * 25
x0 = np.random.randn(500, 1)

lamb = 4
r = 5
s = 0.01

def f(x):
    return 0.5 * pow(np.linalg.norm(np.dot(A,x) - b), 2) + lamb * np.linalg.norm(x,ord=1)

def df(x):
    grad = np.dot(A,x) - b
    tmp = np.zeros(x.shape)
    tmp[x<0] = -1.0
    tmp[x>0] = 1.0
    tmp = tmp * lamb

    return tmp

if __name__ == "__main__":

    xs, fs = run_nesterov(f, df, x0, s, r, epsilon=pow(10, -3))

    xs = np.array(xs)
    fs = np.array(fs)

    plt.figure()
    plt.plot(np.arange(len(xs)), fs)
    plt.xlabel("iteration number")
    plt.ylabel("function value")
    plt.title("Nesterov's Lasso")
    plt.show()

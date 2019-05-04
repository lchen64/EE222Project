import numpy as np
from nesterov import run_nesterov
import matplotlib.pyplot as plt

r = 3
s = 1.0
x0 = np.ones((3,1))

def f(x):
    return 2 * 0.01 * pow(x[0], 2) + 5 * 0.001 * pow(x[1], 2) + \
            1 * pow(10, -4) * pow(x[2], 2) 
            # + 5 * pow(10, -5) * pow(x[3], 2)  

def df(x):
    grad = np.zeros(x.shape)
    grad[0] = 4 * 0.01 * x[0]
    grad[1] = 0.01 * x[1]
    grad[2] = 2 * pow(10,-4)*x[2]
    # grad[3] = pow(10,-4) * x[3]

    return grad

if __name__ == "__main__":

    xs, fs = run_nesterov(f, df, x0, s, r, epsilon=pow(10, -3))

    xs = np.array(xs)
    fs = np.array(fs)
    t = np.arange(len(xs))

    k = 0
    t = t[k:]
    fs = fs[k:]

    plt.figure()
    plt.yscale('log',basey=10) 
    plt.plot(t, fs, linewidth=1)
    plt.xlabel("iteration number")
    plt.ylabel("function value")
    plt.title("Nesterov's quadratic function")
    plt.show()

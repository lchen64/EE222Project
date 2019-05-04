import scipy
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt

N = 100 # number of iterations
N_arr = np.arange(N+1)

def run_nesterov(f, df, x0, s, r, epsilon=pow(10, -3)):
    """
    This function runs the nesterov descent algorithm.

    :param f: 
    """
    x_curr = x0
    y_curr = x0
    xs = []
    xs.append(x0)

    fs = []
    f_x = f(x0)
    fs.append(f_x)
    N = 1
    while N < 3000:
        x_prev = x_curr
        x_curr = y_curr - s * df(y_curr)
        y_curr = x_curr + 1.0*(N-1)/(N+r-1)*(x_curr-x_prev)

        xs.append(x_curr)
        fs.append(f(x_curr))
        N += 1

    return xs, fs


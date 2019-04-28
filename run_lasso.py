import numpy as np
from nesterov import run_nesterov
import matplotlib.pyplot as plt

A = np.random.randn(100, 500)
b = np.random.randn(100, 1) * 25
# x0 = np.random.randn(500, 1)
x0 = np.ones((500, 1))
lamb = 4
r = 4

ATA = np.dot(A.T, A)
eigs, _ = np.linalg.eig(ATA)
eig_1 = np.max(eigs)
L = eig_1 + 2 * lamb

s = 1.0 / L
print(eig_1)
# s = 0.000002
print(s)

def f(x):
    return 0.5 * pow(np.linalg.norm(np.dot(A,x) - b), 2) + lamb * np.linalg.norm(x,ord=1)

def df(x):
    grad = np.dot(A.T, np.dot(A,x) - b)
    tmp = np.zeros(x.shape)
    tmp[x<0] = -1.0
    tmp[x>0] = 1.0
    tmp = tmp * lamb

    return grad + tmp

if __name__ == "__main__":

    xs, fs = run_nesterov(f, df, x0, s, r, epsilon=pow(10, -3))

    xs = np.array(xs)
    fs = np.array(fs)
    t = np.arange(len(xs))

    k = 20
    t = t[k:]
    fs = fs[k:]

    plt.figure()
    plt.yscale('log',basey=10) 
    plt.plot(t, fs, linewidth=1)
    plt.xlabel("iteration number")
    plt.ylabel("function value")
    plt.title("Nesterov's Lasso")
    plt.show()

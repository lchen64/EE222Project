import numpy as np
from nesterov import run_nesterov
import matplotlib.pyplot as plt
import cvxpy as cp

# A = np.random.randn(500, 500)
# b = np.random.randn(500, 1) * 3
N = 500
A = np.random.randn(N, N)
b = np.random.randn(N, 1) * 0.0003
# x0 = np.ones((500, 1)) * 2
# x0 = np.random.randn(500, 1) * 2
x0 = np.random.uniform(low=-0.5, high=0.5, size=b.shape)
lamb = 4

ATA = np.dot(A.T, A)
eigs, _ = np.linalg.eig(ATA)
eig_1 = np.max(eigs)
L = eig_1 + lamb

# s = 1.0 / L
print(eig_1)
s = 0.000002
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

def objective_fn(X, Y, beta, lambd):
    return 0.5 * cp.norm(cp.matmul(X, beta) - Y, 2)**2 + lambd * cp.norm(beta, 1)

def f_star():
    beta = cp.Variable(A.shape[1])
    lambd = cp.Parameter(nonneg=True)
    lambd.value = lamb
    problem = cp.Problem(cp.Minimize(objective_fn(A, np.reshape(b, (len(b),)), beta, lambd)))

    problem.solve()
    beta = beta.value
    beta = np.reshape(beta, (len(beta), 1))
    return f(beta), beta

if __name__ == "__main__":
    f_opt, x_opt = f_star()
    print(f_opt)

    # x0 = x_opt + np.random.randn(x_opt.shape[0], x_opt.shape[1]) * 0.0001

    rs = [3, 4, 5]

    plt.figure()

    lines = []
    plt.yscale('log',basey=10) 
    for i in range(len(rs)):
        r = rs[i]
        _, fs = run_nesterov(f, df, x0, s, r, epsilon=pow(10, -3))
        fs = np.array(fs) - f_opt
        t = np.arange(len(fs))
        line, = plt.plot(t, fs, linewidth=1, label="r = {}".format(r))
        lines.append(line)

    plt.legend(lines)
    # plt.yscale('log',basey=10) 
    plt.xlabel("iteration")
    plt.ylabel("f - f*")
    plt.title("Least square with l-1 regularization, N = {}".format(N))
    plt.savefig("plots/Lasso_N_{}.png".format(N))
    plt.show()

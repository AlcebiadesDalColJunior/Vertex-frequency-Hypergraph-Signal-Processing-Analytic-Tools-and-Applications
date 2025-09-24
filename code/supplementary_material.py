
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

x = range(7)

delta = np.array([1., 0., 0., 0., 0., 0., 0.])

xticklabels = range(1,7+1)

x_smooth_05 = np.array([0.70778249,0.13635384,0.11034135,0.0212766,0.01731811,0.00346365,0.00346365])

x_smooth_1 = np.array([0.57985825,0.17985548,0.13958492,0.04347816,0.0343228,0.01144207,0.01144207])

plt.figure()
plt.plot(x,delta,c='b', label=r'$x$')
plt.plot(x,x_smooth_05,c='r', label=r'$x_0$, $\gamma=0.5$')
plt.plot(x,x_smooth_1,c='g', label=r'$x_0$, $\gamma=1$')
plt.legend()
plt.xticks(x,xticklabels)
plt.savefig("results/path_hypergraph7_x_smooth.pdf", bbox_inches='tight')
plt.show()


x_smooth_05_7b = np.array([0.62896458,0.21517172,0.11034129,0.02127659,0.01731814,0.00346365,0.00346365])

x_smooth_1_7b = np.array([0.51028614,0.24942603,0.13958213,0.04347762,0.03432384,0.01144221,0.01144221])

plt.figure()
plt.plot(x,delta,c='b', label=r'$x$')
plt.plot(x,x_smooth_05_7b,c='r', label=r'$x_0$, $\gamma=0.5$')
plt.plot(x,x_smooth_1_7b,c='g', label=r'$x_0$, $\gamma=1$')
plt.legend()
plt.xticks(x,xticklabels)
plt.savefig("results/path_hypergraph7b_x_smooth.pdf", bbox_inches='tight')
plt.show()
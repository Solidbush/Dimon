import numpy as np  #для работы с массивами данных
import matplotlib.pyplot as plt  #для работы с графиками


np.random.seed(0)
x = np.random.randn(512, 2)
y = np.logical_xor(x[:, 0] > 0, x[:,1] > 0)
y = np.where(y, 1, -1)
plt.figure(1)
plt.scatter(x[y == 1, 0], x[y == 1, 1], c='b', marker='x', label='1')
plt.scatter(x[y == -1, 0], x[y == -1, 1], c='r', marker='s', label='-1')
plt.ylim(-3.0, 3.0); plt.xlim(-3.0, 3.0)
plt.legend()
plt.title("Исходные данные")
plt.show()

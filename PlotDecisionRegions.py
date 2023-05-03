# для работы с массивами данных
import numpy as np
# для работы с графиками
import matplotlib.pyplot as plt
# для создания цветовой карты
from matplotlib.colors import ListedColormap


def plot_decision_regions(x, y, classifier, resolution=0.02, test_idx=None):
    # обозначение образцов маркерами
    markers = ('s', 'x', 'o', '^', 'v')
    # настройка цветовой палитры
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    # Построение графика поверхности решения
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = x[:, 0].min()-1, x[:, 0].max()+1
    x2_min, x2_max = x[:, 1].min()-1, x[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())  # настройка границ графика
    plt.ylim(xx2.min(), xx2.max())
    # отметить образцы классов
    for idx, cl in enumerate(np.unique(y)):
        # перебор элементов,отслеживая индекс
        plt.scatter(x=x[y == cl, 0],
                    y=x[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    edgecolor='black',
                    marker=markers[idx],
                    label=cl)
        if test_idx:
            # обозначение тестовых образцов
            x_test = x[test_idx, :]
            plt.scatter(x_test[:, 0],
                        x_test[:, 1],
                        c='w', alpha=0.3,
                        edgecolor='black',
                        linewidths=1, marker='o',
                        s=120, label='test set')

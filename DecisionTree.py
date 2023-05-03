# Исходные данные
from InputData import *

# Pазделение данных на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Cоздание и тренировка модели на основе дерева решений
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=123)
tree.fit(X_train, y_train)

# Ошибки обучения на обучающей и тестовой выборках
err_train = np.mean(y_train != tree.predict(X_train))
err_test = np.mean(y_test != tree.predict(X_test))
print("\n err_train = ", err_train,
"\n err_test = ", err_test)

#Построение графика
from PlotDecisionRegions import plot_decision_regions
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, test_idx=range(105, 150), classifier=tree,)
plt.xlabel('')
plt.ylabel('')
plt.legend(loc='upper left')
plt.show()

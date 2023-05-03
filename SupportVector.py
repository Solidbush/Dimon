# Исходные данные
from InputData import *
 # Pазделение данных на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)

#Стандартизация
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#Тренировка и подбор наилучших параметров
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = [{'C': np.logspace(0, 10, num=1),
               'gamma': np.logspace(0, 10, num=1),
               'kernel': ['rbf']}]
svm7 = SVC()
grid = GridSearchCV(svm7, param_grid)  #Подбор параметров из списка param_grid
svm7.fit(X_train_std, y_train)
grid.fit(X_train_std, y_train)
cvres = grid.cv_results_
best_params = grid.best_params_  #Вывод лучших параметров
print('\n best_params=', best_params)
print('CV best score = ', grid.best_score_)
print('CV error = ', 1 - grid.best_score_)
print('best C = ', grid.best_estimator_.C)
print('best gamma = ', grid.best_estimator_.gamma)

#Обучение оптимизированной модели
svm_best = grid.best_estimator_
print("Модель bestSVM:",
      ": kernel=", svm_best.kernel,
      "; C=", svm_best.C,
      "; gamma=", svm_best.gamma)
svm_best.fit(X_train, y_train)
# Ошибки обучения на обучающей и тестовой выборках
err_train = np.mean(y_train != svm7.predict(X_train_std))
err_test = np.mean(y_test != svm7.predict(X_test_std))
print("\n\n Модель SVM7:",
 "\n err_train = %.4f" % err_train,
 "\n err_test = %.4f" % err_test)
# Построение графика области решений
from PlotDecisionRegions import plot_decision_regions
X_combined = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plt.figure(figsize=(12, 8))
plot_decision_regions(X_combined,y_combined, classifier=svm_best)
plt.show()

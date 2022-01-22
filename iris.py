import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


iris = load_iris()
#print(iris.feature_names)
#print(iris.target_names)
#print(iris.data)
iris_data = pd.DataFrame(iris.data, columns= iris.feature_names)
iris_data['target'] = iris.target
print(iris_data.head())

#sns.pairplot(iris_data, hue = "target")
#plt.show()

x = iris_data.iloc[:, 2:-1].values
y = iris_data.iloc[:, -1].values
#plt.scatter(x[:, 0], x[:, 1], c=y)
#plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=10)
k_range = np.arange(1,20)
train_score_list = []
test_score_list = []
"""
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred_train = knn.predict(x_train)
    y_pred_test = knn.predict(x_test)
    train_score_list.append(accuracy_score(y_train, y_pred_train))
    test_score_list.append(accuracy_score(y_test, y_pred_test))
plt.plot(k_range, train_score_list, color = 'r', label="training accuracy")
plt.plot(k_range,test_score_list,color='b', label="testing accuracy")
plt.xlabel("Value of k of KNN")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
"""

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

y_pred_test = knn.predict(x_test)
y_pred_train = knn.predict(x_train)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_train, y_pred_train))
print(accuracy_score(y_test, y_pred_test))

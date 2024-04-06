import pandas
import matplotlib.pyplot as plt
import numpy as np

location = ("C:/Users/aref/Desktop/mehdiroozitalab-knn/datas/iris.data")
iris = pandas.read_csv(location, sep = "," )

#print(iris.head)

iris.columns = [
    "sepal length",
    "sepal width",
    "petal length",
    "petal width",
    "class",
]

iris_temp = iris.drop("class", axis=1)
iris_temp = iris_temp.drop("petal length",axis=1)
iris_temp = iris_temp.drop("petal width",axis=1)

X = iris_temp
X = X.values
y = iris["class"]
y = y.values

new_data_point = np.array([
    5.8,
    2.9,
])

distances = np.linalg.norm(X - new_data_point, axis=1)

k = 3
nearest_neighbor = distances.argsort()[:k]
print(nearest_neighbor)
nearest_neighbor_class = y[nearest_neighbor]
print(nearest_neighbor_class)


k = 5
nearest_neighbor = distances.argsort()[:k]
print(nearest_neighbor)
nearest_neighbor_class = y[nearest_neighbor]
print(nearest_neighbor_class)

import matplotlib.pyplot as plt

for name, group in iris.groupby("class"):   
    plt.scatter(group["sepal length"], group["sepal width"], label=name)

plt.scatter(new_data_point[0], new_data_point[1], label="new-data")

plt.legend()
plt.show()


# iris_temp["petal width"].hist(bins = 15)
# plt.show()

#correlation_matrix = iris_temp.corr()
#print(correlation_matrix["petal width"])
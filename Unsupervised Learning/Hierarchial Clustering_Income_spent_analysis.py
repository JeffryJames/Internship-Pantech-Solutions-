import pandas as pd

data = pd.read_csv("C:/Users/Jeffr/Desktop/Intern - Pantech Solutions/Days/Day 20/20_ClusterringIncomeSpentusingHierarchialClusterring/dataset.csv")
data

data = data.drop(["CustomerID"],axis=1)
print(data.shape)
print(data.head())
print(data.describe())

print(data.isnull().sum())

#Label Encoding

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data["Gender"] = label_encoder.fit_transform(data["Gender"])

print(data["Gender"])

from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="average")
y_means = model.fit_predict(data)
y_means

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as cluster

plt.figure(1, figsize=(16,8))
dendrogram = cluster.dendrogram(cluster.linkage(data, method="ward"))

plt.title("Dendrogram Tree Graph")
plt.xlabel("Customers")
plt.ylabel("Distances")
plt.show()


X = data.iloc[:,[2,3]].values
X


plt.scatter(X[y_means==0,0], X[y_means==0,1], s=50, c="purple", label = "Cluster 1")
plt.scatter(X[y_means==1,0], X[y_means==1,1], s=50, c="blue", label = "Cluster 2")
plt.scatter(X[y_means==2,0], X[y_means==2,1], s=50, c="red", label = "Cluster 3")
plt.scatter(X[y_means==3,0], X[y_means==3,1], s=50, c="yellow", label = "Cluster 4")
plt.scatter(X[y_means==4,0], X[y_means==4,1], s=50, c="cyan", label = "Cluster 5")

plt.title("Income Spent Analysis - Hierarchial Clustering")
plt.xlabel("Income")
plt.ylabel("Spent")
plt.show()





from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Standardize
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# PCA
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.components_)
X = pca.transform(X)

# Color definitions
colors = {
  0: '#b40426',
  1: '#3b4cc0',
  2: '#f2da0a',
}

# Make plot of projection
colors = list(map(lambda x: colors[x], y))
plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.title(f'Visualizing the principal components with Scikit-learn based PCA')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.show()

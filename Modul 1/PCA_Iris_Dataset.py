from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt #untuk visualisasi data

#Load dataset Iris (4 fitur -> direduksi menjadi 2d)
data = load_iris()
X = data.data

#Reduksi dimensi dengan PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

#Visualidasi
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data.target)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA pada Dataset Iris")
plt.show()

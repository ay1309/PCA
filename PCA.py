import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data  # solo se suman las caracteristicas

media = np.mean(X, axis=0)
normalizar = X - media # este proceso es para normalizar los valores

matriz = np.cov(normalizar, rowvar=False)
print("Matriz Covarianza:")
print(matriz)   # calcular y mostrar matriz de convarianza

autovalores, autovectores = np.linalg.eigh(matriz)

valoresOrdenados = np.argsort(autovalores)[::-1]
autovalores = autovalores[valoresOrdenados]
autovectores = autovectores[:, valoresOrdenados]

W = autovectores[:, :2]

X_pca = np.dot(normalizar, W)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap='viridis')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.title('PCA')
plt.colorbar()
plt.show()

pip install pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Carregar o dataset
url = "https://raw.githubusercontent.com/saurabh319/Data-Science-Projects/master/Mall_Customers.csv"
df = pd.read_csv(url)

# Visualizar as primeiras linhas
print(df.head())

# Selecionar apenas as colunas de interesse para clusterização
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar KMeans e calcular a inércia para diferentes números de clusters
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plotar a Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.xlabel("Número de Clusters")
plt.ylabel("Inércia")
plt.title("Elbow Curve para Determinação do Número de Clusters")
plt.show()

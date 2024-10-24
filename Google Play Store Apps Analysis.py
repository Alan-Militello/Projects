!pip install kaggle pandas seaborn matplotlib

import zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Baixar dataset do Kaggle
!kaggle datasets download -d lava18/google-play-store-apps

# Descompactar arquivo
with zipfile.ZipFile('google-play-store-apps.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Carregar dados
df = pd.read_csv('googleplaystore.csv')

# Exibir primeiras linhas
print(df.head())

# Analisar distribuição de classificações
plt.figure(figsize=(10,6))
sns.histplot(df['Rating'], bins=20, kde=True)
plt.title('Distribuição de Classificações dos Aplicativos')
plt.show()

# Top 10 aplicativos mais baixados
df['Installs'] = df['Installs'].str.replace(',', '').str.replace('+', '').astype(int)
top_10_installed = df.nlargest(10, 'Installs')[['App', 'Installs']]
plt.figure(figsize=(10,6))
sns.barplot(x='Installs', y='App', data=top_10_installed)
plt.title('Top 10 Aplicativos Mais Baixados na Google Play Store')
plt.show()

# Análise de classificações por categoria
plt.figure(figsize=(12,6))
sns.boxplot(x='Category', y='Rating', data=df)
plt.title('Classificações por Categoria de Aplicativo')
plt.xticks(rotation=90)
plt.show()

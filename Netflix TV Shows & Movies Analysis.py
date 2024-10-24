!pip install kaggle pandas seaborn matplotlib

import zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Baixar dataset do Kaggle
!kaggle datasets download -d shivamb/netflix-shows

# Descompactar arquivo
with zipfile.ZipFile('netflix-shows.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Carregar dados
df = pd.read_csv('netflix_titles.csv')

# Visualizar primeiros registros
print(df.head())

# Analisar lançamentos por ano
plt.figure(figsize=(12,6))
sns.countplot(y='release_year', data=df, order=df['release_year'].value_counts().index)
plt.title("Lançamentos por Ano na Netflix")
plt.show()

# Comparar quantidade de filmes e programas de TV
plt.figure(figsize=(6,6))
sns.countplot(x='type', data=df)
plt.title('Filmes vs Programas de TV na Netflix')
plt.show()

# Análise dos principais gêneros
df['listed_in'] = df['listed_in'].apply(lambda x: x.split(',')[0])
plt.figure(figsize=(10,6))
sns.countplot(y='listed_in', data=df, order=df['listed_in'].value_counts().index[:10])
plt.title('Principais Gêneros de Filmes e Séries')
plt.show()

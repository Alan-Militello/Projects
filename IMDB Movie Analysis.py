!pip install kaggle pandas seaborn matplotlib

import zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Baixar dataset do Kaggle
!kaggle datasets download -d stefanoleone992/imdb-extensive-dataset

# Descompactar arquivo
with zipfile.ZipFile('imdb-extensive-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Carregar dados
df = pd.read_csv('IMDb_movies.csv')

# Estatísticas básicas
print(df.head())

# Top 10 filmes por receita
top_movies = df.nlargest(10, 'revenue')
sns.barplot(x='title', y='revenue', data=top_movies)
plt.title("Top 10 Filmes por Receita")
plt.xticks(rotation=90)
plt.show()

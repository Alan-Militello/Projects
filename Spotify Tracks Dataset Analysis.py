!pip install kaggle pandas seaborn matplotlib

import zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Baixar dataset do Kaggle
!kaggle datasets download -d zaheenhamidani/ultimate-spotify-tracks-db

# Descompactar arquivo
with zipfile.ZipFile('ultimate-spotify-tracks-db.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Carregar dados
df = pd.read_csv('tracks.csv')

# Exibir primeiras linhas
print(df.head())

# Analisar a distribuição de popularidade
plt.figure(figsize=(10,6))
sns.histplot(df['popularity'], kde=True)
plt.title('Distribuição de Popularidade das Músicas no Spotify')
plt.show()

# Análise de dança e energia em músicas populares
plt.figure(figsize=(10,6))
sns.scatterplot(x='danceability', y='energy', hue='popularity', size='popularity', data=df)
plt.title('Relação entre Dança, Energia e Popularidade')
plt.show()

# Top 10 músicas mais populares
top_10 = df.nlargest(10, 'popularity')[['name', 'popularity', 'artists']]
print(top_10)

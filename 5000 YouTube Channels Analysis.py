!pip install kaggle pandas seaborn matplotlib

import zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Baixar dataset do Kaggle
!kaggle datasets download -d datasnaek/youtube-new

# Descompactar arquivo
with zipfile.ZipFile('youtube-new.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Carregar dados
df = pd.read_csv('CAvideos.csv')

# Estatísticas básicas
print(df.describe())

# Análise de vídeos com mais visualizações
top_videos = df.nlargest(10, 'views')
sns.barplot(x='title', y='views', data=top_videos)
plt.title("Top 10 Vídeos com Mais Visualizações")
plt.xticks(rotation=90)
plt.show()

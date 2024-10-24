!pip install kaggle pandas seaborn matplotlib

import zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Baixar dataset do Kaggle
!kaggle datasets download -d carrie1/ecommerce-data

# Descompactar arquivo
with zipfile.ZipFile('ecommerce-data.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Carregar dados
df = pd.read_csv('data.csv')

# Análise de compras
sns.histplot(df['Quantity'])
plt.title('Distribuição de Quantidade de Itens Comprados')
plt.show()

# Análise de receita por país
sns.barplot(x='Country', y='Revenue', data=df)
plt.title('Receita por País')
plt.show()

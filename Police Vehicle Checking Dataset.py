!pip install kaggle pandas seaborn matplotlib

import zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Baixar dataset do Kaggle
!kaggle datasets download -d kwullum/traffic-violations

# Descompactar arquivo
with zipfile.ZipFile('traffic-violations.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Carregar dados
df = pd.read_csv('Traffic_Violations.csv')

# Exibir as primeiras linhas
print(df.head())

# Analisar motivos das paradas de veículos
plt.figure(figsize=(12,6))
sns.countplot(y='Violation Type', data=df, order=df['Violation Type'].value_counts().index)
plt.title("Motivos das Paradas de Veículos")
plt.show()

# Taxas de busca por etnia
plt.figure(figsize=(12,6))
sns.countplot(x='Race', hue='Search Conducted', data=df)
plt.title('Taxa de Busca por Etnia')
plt.show()

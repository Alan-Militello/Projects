!pip install kaggle pandas seaborn matplotlib

import zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Baixar dataset do Kaggle
!kaggle datasets download -d uciml/adult-census-income

# Descompactar arquivo
with zipfile.ZipFile('adult-census-income.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Carregar dados
df = pd.read_csv('adult.csv')

# Visualizar dados
print(df.head())

# Distribuição de renda por nível educacional
sns.countplot(x='education', hue='income', data=df)
plt.title("Renda por Educação")
plt.xticks(rotation=90)
plt.show()
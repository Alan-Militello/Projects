!pip install kaggle pandas seaborn matplotlib

import zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Baixar dataset do Kaggle
!kaggle datasets download -d kaggle/sf-salaries

# Descompactar arquivo
with zipfile.ZipFile('sf-salaries.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Carregar dados
df = pd.read_csv('Salaries.csv')

# Exibir primeiras linhas
print(df.head())

# Analisar distribuição de salários
plt.figure(figsize=(10,6))
sns.histplot(df['BasePay'], kde=True)
plt.title('Distribuição de Salários Base')
plt.show()

# Analisar os maiores salários
top_10_jobs = df.groupby('JobTitle')['TotalPayBenefits'].mean().nlargest(10).reset_index()
plt.figure(figsize=(10,6))
sns.barplot(x='TotalPayBenefits', y='JobTitle', data=top_10_jobs)
plt.title('Top 10 Cargos com Maiores Salários')
plt.show()

# Analisar o uso de overtime por departamento
plt.figure(figsize=(10,6))
sns.barplot(x='OvertimePay', y='JobTitle', data=df.groupby('JobTitle')['OvertimePay'].sum().reset_index().nlargest(10, 'OvertimePay'))
plt.title('Overtime por Departamento')
plt.show()

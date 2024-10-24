!pip install kaggle pandas seaborn matplotlib

import zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Baixar dataset do Kaggle
!kaggle datasets download -d akram24/amazon-sales-dataset

# Descompactar arquivo
with zipfile.ZipFile('amazon-sales-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Carregar dados
df = pd.read_csv('Amazon Sales Report.csv')

# Exibir primeiras linhas
print(df.head())

# Top 10 produtos mais vendidos
top_10_products = df.groupby('Product')['Sales'].sum().nlargest(10).reset_index()
plt.figure(figsize=(10,6))
sns.barplot(x='Sales', y='Product', data=top_10_products)
plt.title('Top 10 Produtos Mais Vendidos')
plt.show()

# Analisar vendas por mês
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Month'] = df['Order Date'].dt.month
monthly_sales = df.groupby('Month')['Sales'].sum().reset_index()
plt.figure(figsize=(10,6))
sns.lineplot(x='Month', y='Sales', data=monthly_sales)
plt.title('Vendas Mensais')
plt.show()

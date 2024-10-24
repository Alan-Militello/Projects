!pip install kaggle pandas seaborn matplotlib

import os
import zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Baixar o dataset do Kaggle
!kaggle datasets download -d andrewmvd/udemy-courses

# Descompactar o arquivo baixado
with zipfile.ZipFile('udemy-courses.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Carregar os dados
df = pd.read_csv('udemy_courses.csv')

# Exibir as primeiras linhas para verificar a estrutura dos dados
print(df.head())

# Informações gerais do dataset
print(df.info())

# Análise de cursos mais populares
top_courses = df.nlargest(10, 'num_subscribers')
plt.figure(figsize=(10,6))
sns.barplot(x='num_subscribers', y='course_title', data=top_courses)
plt.title("Top 10 Cursos Mais Populares da Udemy")
plt.xticks(rotation=90)
plt.show()

# Análise da distribuição de preços dos cursos
plt.figure(figsize=(10,6))
sns.histplot(df['price'], bins=20)
plt.title("Distribuição dos Preços dos Cursos Udemy")
plt.show()

# Análise da classificação média por nível de curso
plt.figure(figsize=(10,6))
sns.barplot(x='level', y='avg_rating', data=df)
plt.title("Classificação Média por Nível de Curso")
plt.show()

# Análise de cursos por categoria
plt.figure(figsize=(12,6))
sns.countplot(y='category', data=df, order=df['category'].value_counts().index)
plt.title("Distribuição de Cursos por Categoria")
plt.show()

# Analisar a relação entre o preço e o número de inscritos
plt.figure(figsize=(10,6))
sns.scatterplot(x='price', y='num_subscribers', data=df)
plt.title("Relação entre Preço e Número de Inscritos")
plt.show()

# Análise de cursos gratuitos e pagos
df['is_paid'] = df['is_paid'].apply(lambda x: 'Paid' if x == True else 'Free')
plt.figure(figsize=(8,5))
sns.countplot(x='is_paid', data=df)
plt.title('Cursos Gratuitos vs Pagos')
plt.show()

# Comparação da classificação média entre cursos gratuitos e pagos
plt.figure(figsize=(8,5))
sns.boxplot(x='is_paid', y='avg_rating', data=df)
plt.title('Comparação da Classificação entre Cursos Gratuitos e Pagos')
plt.show()

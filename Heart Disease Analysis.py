!pip install kaggle pandas seaborn matplotlib scikit-learn

import os
import zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Baixar o dataset do Kaggle
!kaggle datasets download -d ronitf/heart-disease-uci

# Descompactar o arquivo baixado
with zipfile.ZipFile('heart-disease-uci.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Renomear o arquivo CSV extraído
os.rename('heart.csv', 'heart.csv')

# Carregar os dados
df = pd.read_csv('heart.csv')

# Exibir os primeiros dados
print(df.head())

# Visualizar distribuição de pacientes com e sem doença cardíaca
sns.countplot(df['target'])
plt.title('Distribuição de Pacientes com Doença Cardíaca')
plt.show()

# Modelo de machine learning
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Acurácia: {accuracy}")
print(f"Matriz de Confusão: \n{conf_matrix}")

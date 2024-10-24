import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Dataset fictício
data = {'Cliente': ['Cliente 1', 'Cliente 2', 'Cliente 3', 'Cliente 4', 'Cliente 5'],
        'Idade': [25, 45, 34, 52, 23],
        'Renda': [3500, 7000, 4800, 9100, 3200],
        'Histórico de Crédito': [2, 5, 4, 6, 3],
        'Inadimplente': [0, 0, 1, 0, 1]}  # 0 = não, 1 = sim

df = pd.DataFrame(data)

# Definir variáveis independentes (features) e a variável alvo (target)
X = df[['Idade', 'Renda', 'Histórico de Crédito']]
y = df['Inadimplente']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar um modelo de regressão logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Prever os resultados para o conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Acurácia: {accuracy * 100:.2f}%")
print("Matriz de confusão:")
print(conf_matrix)

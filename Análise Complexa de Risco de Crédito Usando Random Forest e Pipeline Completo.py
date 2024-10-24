import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Dataset fictício com mais detalhes
data = {'Cliente': ['Cliente 1', 'Cliente 2', 'Cliente 3', 'Cliente 4', 'Cliente 5', 'Cliente 6'],
        'Idade': [25, 45, 34, 52, 23, 40],
        'Renda': [3500, 7000, 4800, 9100, 3200, 6200],
        'Histórico de Crédito': [2, 5, 4, 6, 3, 7],
        'Inadimplente': [0, 0, 1, 0, 1, 0]}

df = pd.DataFrame(data)

# Definir variáveis independentes e alvo
X = df[['Idade', 'Renda', 'Histórico de Crédito']]
y = df['Inadimplente']

# Criar pipeline de machine learning
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normalizar os dados
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar o modelo
pipeline.fit(X_train, y_train)

# Avaliar o modelo com validação cruzada
scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f"Acurácia média na validação cruzada: {scores.mean() * 100:.2f}%")

# Prever e avaliar o modelo
y_pred = pipeline.predict(X_test)
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

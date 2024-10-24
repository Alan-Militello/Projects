import pandas as pd

# Criar um dataset simples
data = {'Cliente': ['Cliente 1', 'Cliente 2', 'Cliente 3', 'Cliente 4', 'Cliente 5'],
        'Idade': [25, 45, 34, 52, 23],
        'Renda': [3500, 7000, 4800, 9100, 3200],
        'Histórico de Crédito': [2, 5, 4, 6, 3]} # Anos de histórico de crédito

df = pd.DataFrame(data)

# Definir uma função de score simples
def calcular_score(id, renda, historico):
    score = (renda * 0.3) + (historico * 100) - (id * 1.5)
    return score

# Aplicar a função de score aos dados
df['Score'] = df.apply(lambda row: calcular_score(row['Idade'], row['Renda'], row['Histórico de Crédito']), axis=1)

# Exibir os resultados
print(df)

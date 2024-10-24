!pip install pandas seaborn matplotlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar dados
df = pd.read_csv('student-mat.csv')

# Exibir dados
print(df.head())

# Comparar notas finais por tempo de estudo
sns.scatterplot(x='studytime', y='G3', data=df)
plt.title("Relação entre Tempo de Estudo e Notas Finais")
plt.show()

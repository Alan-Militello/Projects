!pip install pandas seaborn matplotlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar dados da COVID-19
url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
df = pd.read_csv(url)

# Analisar casos diários no Brasil
brasil_df = df[df['location'] == 'Brazil']
sns.lineplot(x='date', y='new_cases', data=brasil_df)
plt.title("Casos Diários no Brasil")
plt.xticks(rotation=90)
plt.show()

#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tweepy')
get_ipython().system('pip install nltk')
get_ipython().system('pip install numpy')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install matplotlib')


import tweepy
import re
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import seaborn as sns
import matplotlib.pyplot as plt

# Configurações de autenticação usando apenas o Bearer Token para a API v2 do Twitter
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAI3NwgEAAAAAcm9fuNgwerrrC%2BF0XMy4QY85PPM%3DQr47RT8WzEnUyrJs5Nx0GxK6MpqklUTGFxqJAo4pplCvf6EfXM'
client = tweepy.Client(bearer_token=bearer_token)

# Baixando recursos do NLTK necessários para pré-processamento
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Coletar tweets usando o Bearer Token
query = "feliz lang:pt"
response = client.search_recent_tweets(query=query, max_results=100, tweet_fields=['created_at', 'lang'])
dados = [(tweet.text, tweet.created_at) for tweet in response.data]

# Função de pré-processamento de texto
def preprocess_text(text):
    text = re.sub(r"http\S+|@\S+|#\S+|[0-9]+", "", text)  # Remove URLs, menções, hashtags e números
    tokens = nltk.word_tokenize(text.lower())  # Tokenização e conversão para minúsculas
    tokens = [t for t in tokens if t not in stopwords.words('portuguese')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]  # Lematização
    return " ".join(tokens)

# Pré-processar todos os tweets coletados
dados_limpos = [preprocess_text(texto) for texto, _ in dados]

# Configuração do modelo de deep learning com Keras
vocab_size = 5000
embedding_dim = 64
lstm_units = 128

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=100),
    LSTM(lstm_units),
    Dense(1, activation='sigmoid')  # Ativação sigmoid para classificação binária
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Conversão dos dados para arrays e divisão em treino e teste
X = np.array(dados_limpos)
y = np.array([1 if "feliz" in texto else 0 for texto in dados_limpos])  # Sentimento simplificado como exemplo

# Divisão dos dados
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Treinamento do modelo
model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val), batch_size=32)

# Avaliação do modelo
y_pred = model.predict(X_val) > 0.5
print("Acurácia:", accuracy_score(y_val, y_pred))
print("Matriz de Confusão:\n", confusion_matrix(y_val, y_pred))

# Exibição da matriz de confusão
conf_matrix = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predição")
plt.ylabel("Verdadeiro")
plt.title("Matriz de Confusão para a Classificação de Sentimento")
plt.show()


# In[3]:


conda create -n sentiment_env python=3.8
conda activate sentiment_env
pip install tweepy nltk numpy scikit-learn tensorflow seaborn matplotlib


# In[4]:


python -c "import platform; print(platform.architecture())"


# In[9]:





# In[ ]:





# In[ ]:





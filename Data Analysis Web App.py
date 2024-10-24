!pip install streamlit pandas matplotlib

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Upload de arquivo CSV
uploaded_file = st.file_uploader("Upload seu arquivo CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # Exibir gráfico de barras
    st.bar_chart(df['column_name'])

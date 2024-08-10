import streamlit as st
import openai
import pandas as pd
import numpy as np

openai.api_key = st.secrets["mykey"]

# Load the CSV file into a Pandas DataFrame
file_path = 'qa_dataset_with_embeddings.csv'
df = pd.read_csv(file_path)

# Convert the Question_Embedding column from string to numpy array
df['Question_Embedding'] = df['Question_Embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))

# Display the first few rows of the DataFrame to verify
df.head()



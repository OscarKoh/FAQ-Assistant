import streamlit as st
import openai
import pandas as pd
import numpy as np
import os

# Set your OpenAI API key
openai.api_key = st.secrets["mykey"]

# Check if the file exists
file_path = 'qa_dataset_with_embeddings.csv'
if os.path.exists(file_path):
    st.write(f"File '{file_path}' found.")
    df = pd.read_csv(file_path)
else:
    st.error(f"File '{file_path}' not found. Please check the file path.")

# Convert the Question_Embedding column from string to numpy array if file loaded successfully
if not df.empty:
    df['Question_Embedding'] = df['Question_Embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
    st.write("Preview of the dataset:")
    st.dataframe(df.head())
else:
    st.error("The dataset is empty or could not be loaded.")

def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(
      input=text,
      model=model
    )
    embedding = response['data'][0]['embedding']
    return embedding

# Streamlit application
def main():
    st.title("Health Q&A Embedding Generator")
    
    # Input field for the user question
    user_question = st.text_input("Enter your question:", "")

    if st.button("Generate Embedding"):
        if user_question:
            # Generate the embedding for the new question
            embedding = get_embedding(user_question)

            # Display the embedding
            st.write("Generated Embedding:")
            st.write(embedding)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()

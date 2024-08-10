import streamlit as st
import openai
import pandas as pd
import numpy as np
import os

# Set your OpenAI API key
openai.api_key = st.secrets["mykey"]

# Check if the file exists and load the dataset without showing messages
file_path = 'qa_dataset_with_embeddings.csv'
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    st.error(f"File '{file_path}' not found. Please check the file path.")

# Convert the Question_Embedding column from string to numpy array if file loaded successfully
if not df.empty:
    df['Question_Embedding'] = df['Question_Embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
else:
    st.error("The dataset is empty or could not be loaded.")

# Function to get embedding from OpenAI
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(
      input=text,
      model=model
    )
    embedding = response['data'][0]['embedding']
    return embedding

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Streamlit application
def main():
    st.title("Health Q&A Embedding Generator")

    # Input field for the user question
    user_question = st.text_input("Enter your question:", "")

    # Clear button
    if st.button("Clear"):
        user_question = ""
        st.experimental_rerun()

    if st.button("Generate Embedding"):
        if user_question:
            # Step 1: Generate the embedding for the new question
            user_embedding = get_embedding(user_question)

            # Step 2: Calculate the cosine similarity between the user's question embedding and all question embeddings in the dataset
            df['Similarity'] = df['Question_Embedding'].apply(
                lambda x: cosine_similarity(x, user_embedding)
            )

            # Step 3: Find the question with the highest similarity score
            most_similar_idx = df['Similarity'].idxmax()
            max_similarity = df.loc[most_similar_idx, 'Similarity']

            # Define a threshold for relevance
            similarity_threshold = 0.85  # You may need to experiment to find the best value

            # Step 4: Check if the highest similarity is above the threshold
            if max_similarity > similarity_threshold:
                st.write("Most relevant question found in the dataset:")
                st.write(f"Question: {df.loc[most_similar_idx, 'Question']}")
                st.write(f"Answer: {df.loc[most_similar_idx, 'Answer']}")
                st.write(f"Similarity Score: {max_similarity:.2f}")

                # Step 5: User rating for helpfulness
                st.write("Was this answer helpful?")
                if st.button("Yes"):
                    st.success("Thank you for your feedback!")
                if st.button("No"):
                    st.warning("Thank you for your feedback! We'll try to improve.")
            else:
                # Step 6: No relevant answer found
                st.warning("I apologize, but I don't have information on that topic yet. Could you please ask other questions?")
        else:
            st.warning("Please enter a question.")

    # FAQs Section
    st.subheader("Common FAQs")
    search_query = st.text_input("Search FAQs:")
    if search_query:
        filtered_df = df[df['Question'].str.contains(search_query, case=False)]
        if not filtered_df.empty:
            st.write("Matching FAQs:")
            for i, row in filtered_df.iterrows():
                st.write(f"Q: {row['Question']}")
        else:
            st.write("No matching FAQs found.")
    else:
        st.write("Frequently Asked Questions:")
        for i, row in df.head(5).iterrows():  # Display the first 5 FAQs
            st.write(f"Q: {row['Question']}")

if __name__ == "__main__":
    main()

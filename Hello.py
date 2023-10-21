import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# Define a function to tokenize text
def tokenize_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    return tokens

# Define a function to generate a bag of words
def bag_of_words(text):
    tokens = tokenize_text(text)
    bow = FreqDist(tokens)
    return bow

# Define a function to create a word frequency bar graph
def create_word_frequency_bar(bow, title):
    plt.figure(figsize=(10, 6))
    bow.plot(20, title=title)
    plt.xticks(rotation=45)
    st.pyplot()

# Define a function to create a word cloud
def create_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    st.image(wordcloud.to_array())
    st.write(title)

# Define a function to get the most similar response
def get_most_similar_response(df, query, top_k=1):
    # Step 1: Prepare Data
    vectorizer = TfidfVectorizer()
    all_data = list(df['Questions']) + [query]

    # Step 2: TF-IDF Vectorization
    tfidf_matrix = vectorizer.fit_transform(all_data)

    # Step 3: Compute Similarity
    document_vectors = tfidf_matrix[:-1]
    query_vector = tfidf_matrix[-1]
    similarity_scores = cosine_similarity(query_vector, document_vectors)

    # Step 4: Sort and Pick Top k Responses
    sorted_indexes = similarity_scores.argsort()[0][-top_k:]
    
    # Fetch the corresponding responses from the DataFrame
    most_similar_responses = df.iloc[sorted_indexes]['Answers'].values
    
    return most_similar_responses

def is_insufficient(prompt):
    return len(prompt.split()) <= 1

# Sample DataFrame with user_chat and response columns
df = pd.read_csv('medicine.csv')

st.title("Welcome to MediBot! 🤖")

description = """
Welcome, I am your trusted companion in the world of medicine and health science. Whether you have questions about the inner workings of the human body, want to understand common medical conditions, or seek information on the latest advancements in healthcare, MediBot is here to assist you. We provide answers to a wide range of questions related to health and medicine. From explaining the function of vital organs to clarifying complex medical procedures, our goal is to empower you with knowledge and information. Ask us anything, and we'll do our best to provide you with accurate and helpful insights into the fascinating world of medicine.
"""

st.markdown(description)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me questions about medicine or health science. (e.g: What is diabetes?)"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    if is_insufficient(prompt):
        insufficient_response = "Insufficient Prompt. Please clarify what you want to know."
        with st.chat_message("assistant"):
            st.markdown(insufficient_response)
        st.session_state.messages.append({"role": "assistant", "content": insufficient_response, "related_query": prompt})
    else:
        # Check if the same prompt was already answered previously
        previous_responses = [m["content"] for m in st.session_state.messages if m["role"] == "assistant" and m["related_query"] == prompt]
        
        if previous_responses:
            for response in previous_responses:
                with st.chat_message("assistant"):
                    st.markdown(response)
        else:
            # Get and display assistant response in chat message container
            responses = get_most_similar_response(df, prompt)
            for response in responses:
                with st.chat_message("assistant"):
                    st.markdown(f"{response}")

            # Add assistant response to chat history
            for response in responses:
                st.session_state.messages.append({"role": "assistant", "content": f"Echo: {response}", "related_query": prompt})
            
            # Create bag of words and word frequency bar graph for user prompt
            user_prompt_bow = bag_of_words(prompt)
            create_word_frequency_bar(user_prompt_bow, "Word Frequency in User Prompt")
            
            # Create bag of words and word frequency bar graph for assistant response
            assistant_response_bow = bag_of_words(" ".join(responses))
            create_word_frequency_bar(assistant_response_bow, "Word Frequency in Assistant Response")
            
            # Create word cloud for user prompt
            create_word_cloud(" ".join(user_prompt_bow), "Word Cloud for User Prompt")
            
            # Create word cloud for assistant response
            create_word_cloud(" ".join(assistant_response_bow), "Word Cloud for Assistant Response")

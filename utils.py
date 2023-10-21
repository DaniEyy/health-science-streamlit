import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import matplotlib.pyplot as plt
import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud import WordCloud
import spacy
import re
import en_core_web_sm
import spacy

# Load the "en_core_web_sm" model
nlp = spacy.load("en_core_web_sm")

def get_most_similar_response(df, query, top_k=1):
    # Step 1: Prepare Data
    vectorizer = TfidfVectorizer()
    all_data = list(df['user_chat']) + [query]

    # Step 2: TF-IDF Vectorization
    tfidf_matrix = vectorizer.fit_transform(all_data)

    # Step 3: Compute Similarity
    document_vectors = tfidf_matrix[:-1]
    query_vector = tfidf_matrix[-1]
    similarity_scores = cosine_similarity(query_vector, document_vectors)

    # Step 4: Sort and Pick Top k Responses
    sorted_indexes = similarity_scores.argsort()[0][-top_k:]
    
    # Fetch the corresponding responses from the DataFrame
    most_similar_responses = df.iloc[sorted_indexes]['response'].values

    highest_score = similarity_scores[0][sorted_indexes[0]]
    print(highest_score)
    if highest_score<=0.3:
        sorry_response = ["I'm sorry, I don't have the answer to that trivia question.","Hmm, I'm not sure about that one.","I don't have that trivia in my database."]
        return [sorry_response[random.randint(0,2)]]
    elif highest_score<=0.7:
        return ["I guess it's "] + most_similar_responses    
    return ["Surely, "] + most_similar_responses

def create_bag_of_words(text):
    # Tokenize the input text
    doc = nlp(treat(text))
    print(doc)

    # Count word frequencies
    word_freq = {}
    for token in doc:
        word = token.text.strip().lower()
        if word and not word.isspace():
            if token.text not in word_freq:
                word_freq[token.text] = 1
            else:
                word_freq[token.text] += 1

    # Sort the dictionary by frequency in descending order
    sorted_word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))

    return sorted_word_freq

def tokenize(text):
	doc = nlp(text)
	return doc

def remove_stop_words(doc):
	filtered_tokens = [token.text for token in doc if not token.is_stop]
	filtered_text = ' '.join(filtered_tokens)
	return (' '.join(filtered_tokens))

def remove_special_keys(filtered_text):
	pattern = r'[^a-zA-z0-9\s(\)\[\]\{\}]'
	cleaned_text = re.sub(pattern, '', filtered_text)
	return cleaned_text.strip(' ')

# For data cleaning
def treat(text):
	return remove_special_keys(remove_stop_words(tokenize(text)))

# Function to generate word frequency bar graph
def generate_word_frequency_bar_graph(str_input, role):
    if str_input=="":
        return

    text = treat(str_input)

    if text:
        word_freq = create_bag_of_words(text)

        # Plot the bag of words
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(word_freq.keys(), word_freq.values())
        ax.set_xlabel('Words')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{role} {role} Bag of Words')
        ax.set_xticklabels(word_freq.keys(), rotation=45, fontsize=8)

        # Show the plot
        st.pyplot(fig)

# Function to generate WordCloud
def generate_wordcloud(str_input, role):
	try:
		str_input = treat(str_input)
		wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(str_input)
		# Plot the WordCloud image
		st.image(wordcloud.to_image(), caption=f"{role} WordCloud")
	except:
		pass

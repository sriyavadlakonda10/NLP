import os
import xml.etree.ElementTree as ET
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import streamlit as st

# Ensure NLTK data is downloaded only if not already available
nltk_data_path = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "nltk_data")
if not os.path.exists(nltk_data_path):
    nltk.download("punkt")
    nltk.download("stopwords")
else:
    nltk.data.path.append(nltk_data_path)

# Function to strip HTML content (if any) using BeautifulSoup
def strip_html(text):
    soup = BeautifulSoup(text, "xml")
    return soup.get_text()

# Function to remove text inside square brackets
def remove_between_square_brackets(text):
    return re.sub(r'\[[^]]*\]', '', text)

# Function to denoise the text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Streamlit app
st.title("NLP Pipeline with Streamlit")

# File upload section
uploaded_file = st.file_uploader("Upload an XML File", type=["xml"])
if uploaded_file is not None:
    # Parse the XML file
    tree = ET.parse(uploaded_file)
    root = tree.getroot()
    root_string = ET.tostring(root, encoding='utf-8').decode('utf-8')

    # Clean the XML string
    sample = denoise_text(root_string)

    # Tokenization
    tokens = word_tokenize(sample)
    sentences = sent_tokenize(sample)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalnum()]

    # Display results
    st.subheader("Tokenized Text")
    st.write(tokens)

    st.subheader("Filtered Tokens (Stopwords Removed)")
    st.write(filtered_tokens)

    st.subheader("Sentences")
    st.write(sentences)

    # Generate WordCloud
    st.subheader("WordCloud")
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(filtered_tokens))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

    # Sentence Scoring using TF-IDF
    st.subheader("TF-IDF Sentence Scoring")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    sentence_scores = tfidf_matrix.sum(axis=1)
    tfidf_scores = {sentences[i]: sentence_scores[i, 0] for i in range(len(sentences))}
    st.write(tfidf_scores)

    # Word Embedding using Word2Vec
    st.subheader("Word2Vec Embeddings")
    word2vec_model = Word2Vec(sentences=[filtered_tokens], vector_size=100, window=5, min_count=1, workers=4)

    word_to_check = st.text_input("Enter a word to check in Word2Vec vocabulary", "data")
    if word_to_check in word2vec_model.wv:
        st.write(f"Word Vector for '{word_to_check}':", word2vec_model.wv[word_to_check])

        similar_words = word2vec_model.wv.most_similar(word_to_check, topn=5)
        st.write(f"Most Similar Words to '{word_to_check}':")
        for word, similarity in similar_words:
            st.write(f"{word}: {similarity:.2f}")
    else:
        st.write(f"Word '{word_to_check}' not found in the Word2Vec vocabulary.")

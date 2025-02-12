import streamlit as st
import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import re
import pickle

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Streamlit Title and Description
st.title("Word2Vec Model Training and Exploration")
st.write("""
This app allows you to train a Word2Vec model on a given paragraph and explore word vectors and their relationships.
""")

# Input Paragraph
st.subheader("Input Paragraph")
paragraph = st.text_area("Enter the paragraph below:", value="""I have three visions for India. In 3000 years of our history, people from all over 
the world have come and invaded us, captured our lands, conquered our minds. From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, 
the British, the French, the Dutch, all of them came and looted us, took over what was ours. Yet we have not done this to any other nation. We have 
not conquered anyone. We have not grabbed their land, their culture, their history and tried to enforce our way of life on them. Why? Because we respect 
the freedom of others. That is why my first vision is that of freedom...""")

# Preprocess the text
def preprocess_text(paragraph):
    text = re.sub(r'\[[0-9]*\]', ' ', paragraph)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    for i in range(len(sentences)):
        sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
    return sentences

# Train Word2Vec Model
def train_word2vec(sentences):
    model = Word2Vec(sentences, min_count=1)
    return model

# Preprocessing and Training
if st.button("Train Word2Vec Model"):
    sentences = preprocess_text(paragraph)
    model = train_word2vec(sentences)
    st.success("Model trained successfully!")
    
    # Save the model using pickle
    with open('word2vec_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    st.info("Model saved as 'word2vec_model.pkl'.")

# Load the saved model
if st.button("Load Saved Model"):
    try:
        with open('word2vec_model.pkl', 'rb') as f:
            model = pickle.load(f)
        st.success("Model loaded successfully!")
    except FileNotFoundError:
        st.error("No saved model found. Please train and save a model first.")

# Word Vector Exploration
if 'model' in locals():
    st.subheader("Word Vector Exploration")
    word = st.text_input("Enter a word to find its vector or similar words:", "war")
    if word in model.wv:
        vector = model.wv[word]
        st.write(f"Vector for the word '{word}':", vector)
        
        similar_words = model.wv.most_similar(word)
        st.write(f"Most similar words to '{word}':")
        for similar_word, similarity in similar_words:
            st.write(f"{similar_word}: {similarity:.2f}")
    else:
        st.error(f"The word '{word}' is not in the vocabulary.")

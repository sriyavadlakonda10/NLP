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

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# Set the directory path to where your XML file is located
os.chdir(r"C:\Users\Sriya v\VS CODE\nlp")

# Parse the XML file
tree = ET.parse("769952.xml")
root = tree.getroot()

# Convert the XML tree to a string for processing
root_string = ET.tostring(root, encoding='utf-8').decode('utf-8')

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

# Clean the root XML string
sample = denoise_text(root_string)

# Tokenization
tokens = word_tokenize(sample)

# Sentence Tokenization
sentences = sent_tokenize(sample)

# Remove stop words
stop_words = set(stopwords.words("english"))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalnum()]

# Generate WordCloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(filtered_tokens))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Sentence Scoring using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
sentence_scores = tfidf_matrix.sum(axis=1)  # Sum TF-IDF scores for each sentence
for i, sentence in enumerate(sentences):
    print(f"Sentence {i + 1}: {sentence} | Score: {sentence_scores[i, 0]:.2f}")

# Word Embedding using Word2Vec
word2vec_model = Word2Vec(sentences=[filtered_tokens], vector_size=100, window=5, min_count=1, workers=4)

# Check if the word "data" exists in the vocabulary
word_to_check = "data"
if word_to_check in word2vec_model.wv:
    print(f"Word Vector for '{word_to_check}':", word2vec_model.wv[word_to_check])
    print("\nMost Similar Words to 'data':")
    similar_words = word2vec_model.wv.most_similar(word_to_check, topn=5)
    for word, similarity in similar_words:
        print(f"{word}: {similarity:.2f}")
else:
    print(f"Word '{word_to_check}' not found in the Word2Vec vocabulary.")

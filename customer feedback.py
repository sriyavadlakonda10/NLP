# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\Sriya v\VS CODE\nlp\customer feedback\Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

# Cleaning the texts
import re
import nltk
# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])  # Keep only letters
    review = review.lower()  # Convert to lowercase
    review = review.split()  # Split into words
    ps = PorterStemmer()
    # Remove stopwords and apply stemming
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)  # Join words back into a single string
    corpus.append(review)

# Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1500)  # You can adjust max_features for better results
X = tfidf.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Training the Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Evaluating the model
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print("\nAccuracy Score:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Checking bias and variance
train_score = classifier.score(X_train, y_train)
test_score = classifier.score(X_test, y_test)

print("\nTraining Accuracy (Bias):", train_score)
print("Testing Accuracy (Variance):", test_score)

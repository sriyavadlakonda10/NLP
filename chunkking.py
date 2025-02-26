import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\Sriya v\OneDrive\Desktop\Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting the dataset into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Function to create and train ANN with different optimizers
def train_ann(optimizer_name):
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=5, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=4, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    
    ann.compile(optimizer=optimizer_name, loss='binary_crossentropy', metrics=['accuracy'])
    ann.fit(X_train, y_train, batch_size=32, epochs=5, verbose=0)
    
    y_pred = ann.predict(X_test)
    y_pred = (y_pred > 0.5)
    
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Optimizer: {optimizer_name}, Accuracy: {accuracy}")
    print("Confusion Matrix:\n", cm)
    print("-" * 50)

# List of optimizers to test
optimizers = ['SGD', 'Adam', 'Adagrad', 'Adadelta', 'Adamax', 'RMSprop']

# Train and evaluate ANN with different optimizers
for opt in optimizers:
    train_ann(opt)



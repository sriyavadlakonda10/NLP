import streamlit as st
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Define the text for the word cloud
text = ("Python Python Python Matplotlib Matplotlib Seaborn Network Plot Violin Chart Pandas "
        "Datascience Wordcloud Spider Radar Parallel Alpha Color Brewer Density Scatter Barplot "
        "Boxplot Violinplot Treemap Stacked Area Chart Visualization Dataviz Donut Pie Time-Series "
        "Wordcloud Sankey Bubble")

# Generate the word cloud
wordcloud = WordCloud(width=480, height=480, margin=0).generate(text)

# Save wordcloud using Pickle
with open("wordcloud.pkl", "wb") as f:
    pickle.dump(wordcloud, f)

# Streamlit UI
st.title("Word Cloud Visualization")

# Load wordcloud from Pickle
with open("wordcloud.pkl", "rb") as f:
    loaded_wordcloud = pickle.load(f)

# Display the word cloud
fig, ax = plt.subplots()
ax.imshow(loaded_wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

import os
import xml.etree.ElementTree as ET
import re
from bs4 import BeautifulSoup

# Set the directory path to where your XML file is located
os.chdir(r"C:\Users\Sriya v\VS CODE\nlp")

# Parse the XML file
tree = ET.parse("769952.xml") 
root = tree.getroot()

# Convert the XML tree to a string for processing
root_string = ET.tostring(root, encoding='utf-8').decode('utf-8')

# Function to strip HTML content (if any) using BeautifulSoup
def strip_html(text):
    # Use the XML parser to ensure XML compatibility
    soup = BeautifulSoup(text, "xml")
    return soup.get_text()

# Function to remove text inside square brackets
def remove_between_square_brackets(text):
    # Use a raw string for the regex pattern to avoid escape sequence warnings
    return re.sub(r'\[[^]]*\]', '', text)

# Function to denoise the text
def denoise_text(text):
    text = strip_html(text)  # Remove any HTML/XML tags
    text = remove_between_square_brackets(text)  # Remove square brackets content
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text

# Clean the root XML string
sample = denoise_text(root_string)

# Print the cleaned text
print(sample)

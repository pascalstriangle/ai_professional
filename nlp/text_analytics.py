# Import necessary libraries
import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')

# %%


# Load the CSV data into a DataFrame
df = pd.read_csv('your_csv_file.csv')

# 1. Clean the text
def clean_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_text'] = df['text_column'].apply(clean_text)

# 2. Create some regex examples (customize as needed)
def apply_regex(text):
    # Example: Extract all email addresses
    emails = re.findall(r'\S+@\S+', text)
    # Example: Extract all phone numbers
    phone_numbers = re.findall(r'\d{3}-\d{3}-\d{4}', text)
    return {
        'emails': emails,
        'phone_numbers': phone_numbers
    }

df['regex_results'] = df['cleaned_text'].apply(apply_regex)

# 3. Tokenization
nlp = spacy.load('en_core_web_sm')

def tokenize_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens

df['tokens'] = df['cleaned_text'].apply(tokenize_text)

# 4. Stop word removal
stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

df['tokens_no_stopwords'] = df['tokens'].apply(remove_stopwords)

# Save the processed DataFrame to a new CSV file if needed
df.to_csv('processed_data.csv', index=False)

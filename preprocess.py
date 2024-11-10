# preprocess.py
import pandas as pd
import re

def load_and_clean_data(file_path):
    conversations = pd.read_csv(file_path, delimiter='\t', header=None, names=['User', 'Response'])
    
    # Basic cleaning: remove any unwanted symbols, lowercase everything
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
        return text
    
    conversations['User'] = conversations['User'].apply(clean_text)
    conversations['Response'] = conversations['Response'].apply(clean_text)
    
    # Dropping any rows with null values (if present)
    conversations.dropna(inplace=True)
    
    return conversations

if __name__ == "__main__":
    file_path = 'data/dialogs.txt'
    conversations = load_and_clean_data(file_path)
    print("Cleaned Data Sample:")
    print(conversations.head())

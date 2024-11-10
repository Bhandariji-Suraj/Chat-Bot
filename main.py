# main.py
import pandas as pd

# Load the dataset
file_path = 'data/dialogs.txt'
conversations = pd.read_csv(file_path, delimiter='\t', header=None, names=['User', 'Response'])

# Basic exploration
print("First few rows of the dataset:")
print(conversations.head())

# Checking the dataset size
print(f"\nTotal number of conversations: {len(conversations)}")

# Count unique responses and unique questions
unique_user_inputs = conversations['User'].nunique()
unique_responses = conversations['Response'].nunique()

print(f"\nUnique user queries: {unique_user_inputs}")
print(f"Unique bot responses: {unique_responses}")

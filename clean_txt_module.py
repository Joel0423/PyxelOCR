import spacy
from nltk.corpus import words
import nltk

# Load English vocabulary
english_vocab = set(words.words())

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    doc = nlp(text)
    cleaned_words = []

    for token in doc:
        # Keep punctuation as it is or keep the word if it's in the vocabulary
        if token.is_punct or token.text.lower() in english_vocab:
            cleaned_words.append(token.text)
    
    # Return text with preserved punctuation
    return "".join([word if word in ".,!?;:'\"()[]{}" else " " + word for word in cleaned_words]).strip()

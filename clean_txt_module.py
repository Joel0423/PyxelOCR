import spacy
from nltk.corpus import words

# Load English vocabulary
english_vocab = set(words.words())

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    return text
    doc = nlp(text)
    cleaned_words = []

    for token in doc:
        # Keep punctuation, numbers, dates, or valid words
        if (token.is_punct or 
            token.text.lower() in english_vocab or 
            token.like_num or 
            token.ent_type_ in ["DATE", "TIME"]):
            cleaned_words.append(token.text)
    
    # Return text with preserved punctuation and spacing
    return "".join([word if word in ".,!?;:'\"()[]{}" else " " + word for word in cleaned_words]).strip()

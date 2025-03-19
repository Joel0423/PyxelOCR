import spacy
from transformers import pipeline
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from clean_txt_module import clean_text

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

class TextSummarizer:
    def __init__(self):
        self.abstractive_summarizer = pipeline("summarization", model="Falconsai/text_summarization")
    
    def extractive_summarize(self, text, length=0.3):
        """Advanced extractive summarization using TF-IDF and TextRank-inspired approach."""
        # Use spaCy for sentence tokenization
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        if not sentences:
            return ""
        
        # Create TF-IDF matrix for sentences
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores based on the sum of their TF-IDF values
        sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
            
        # Normalize scores
        if len(sentence_scores) > 0:
            max_score = max(sentence_scores)
            if max_score > 0:  # Avoid division by zero
                sentence_scores = sentence_scores / max_score
            
        # Select top sentences
        select_length = max(int(len(sentences) * length), 1)
        top_sentences_indices = np.argsort(sentence_scores)[-select_length:]
        top_sentences_indices = sorted(top_sentences_indices)
        
        summary = ' '.join([sentences[i] for i in top_sentences_indices])
        return summary
    
    def abstractive_summarize(self, text, max_length=130, min_length=30):
        """Abstractive summarization using Fine-Tuned T5 Small for Text Summarization."""
        try:
            summary = self.abstractive_summarizer(text, 
                                                max_length=max_length, 
                                                min_length=min_length, 
                                                do_sample=False)[0]['summary_text']
            return summary
        except Exception as e:
            return f"Error in abstractive summarization: {str(e)}"
    
    def hybrid_summarize(self, text, length=0.3):
        """Combines both extractive and abstractive summarization."""
        # First get extractive summary
        text = clean_text(text)
        extractive_summary = self.extractive_summarize(text, length)
        
        # Then apply abstractive summarization on the extractive summary
        try:
            final_summary = self.abstractive_summarize(extractive_summary)
            return {
                'final_summary': final_summary,
                'extractive_summary': extractive_summary
            }
        except:
            return {
                'final_summary': extractive_summary,
                'extractive_summary': extractive_summary
            }
    
    def get_key_phrases(self, text):
        """Extracts key phrases from the text."""
        text = clean_text(text)
        doc = nlp(text)
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        return list(set(noun_phrases))[:min(8,len(noun_phrases))]  # Remove duplicates 
import spacy
from textblob import TextBlob
from collections import defaultdict, Counter

# Load spaCy models
nlp = spacy.load('en_core_web_sm')
nlp_affect = spacy.load('affect_ner')

class SentimentAnalyzer:
    def __init__(self):
        self.affect_percent = {
            'fear': 0.0, 'anger': 0.0, 'anticipation': 0.0, 'trust': 0.0, 
            'surprise': 0.0, 'positive': 0.0, 'negative': 0.0, 'sadness': 0.0, 
            'disgust': 0.0, 'joy': 0.0
        }
    
    def analyze_sentiment_combined(self, text):
        """Analyzes sentiment using TextBlob."""
        blob = TextBlob(text)
        
        # Get overall sentiment
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine sentiment category
        if polarity >= 0.05:
            sentiment = "Positive"
        elif polarity <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        # Get sentence-level analysis
        sentence_sentiments = []
        for sentence in blob.sentences:
            sentence_sentiments.append({
                'text': str(sentence),
                'score': sentence.sentiment.polarity,
                'subjectivity': sentence.sentiment.subjectivity
            })
        
        return {
            'sentiment': sentiment,
            'score': polarity,
            'subjectivity': subjectivity,
            'details': {
                'sentence_sentiments': sentence_sentiments
            }
        }
    
    def get_emotional_tone(self, text):
        """Analyzes emotional tone of the text using affect_ner model."""
        emotions = []
        doc = nlp_affect(text)
        
        if len(doc.ents) != 0:
            for ent in doc.ents:
                emotions.append(ent.label_.lower())
            
            affect_counts = Counter()
            for emotion in emotions:
                affect_counts[emotion] += 1
            
            sum_values = sum(affect_counts.values())
            if sum_values > 0:
                for key in affect_counts.keys():
                    self.affect_percent[key] = float(affect_counts[key]) / float(sum_values)
        
        # Convert percentages to counts for display
        emotion_counts = {
            emotion: int(percentage * 100) 
            for emotion, percentage in self.affect_percent.items() 
            if percentage > 0
        }
        
        return emotion_counts 
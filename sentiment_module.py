import spacy
from textblob import TextBlob
from collections import defaultdict

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Custom sentiment lexicon
SENTIMENT_LEXICON = {
    'positive': {
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'happy',
        'best', 'love', 'beautiful', 'perfect', 'awesome', 'brilliant', 'outstanding',
        'superb', 'delighted', 'pleased', 'joy', 'success', 'successful'
    },
    'negative': {
        'bad', 'terrible', 'awful', 'horrible', 'poor', 'worst', 'hate',
        'disappointing', 'disappointed', 'fail', 'failed', 'failure', 'wrong',
        'mistake', 'error', 'problem', 'difficult', 'impossible', 'angry', 'sad'
    }
}

class SentimentAnalyzer:
    def __init__(self):
        self.emotion_words = {
            'joy': ['happy', 'joyful', 'delighted', 'excited', 'pleased', 'glad', 'cheerful'],
            'sadness': ['sad', 'unhappy', 'depressed', 'gloomy', 'miserable', 'disappointed'],
            'anger': ['angry', 'furious', 'irritated', 'annoyed', 'outraged', 'frustrated'],
            'fear': ['scared', 'afraid', 'fearful', 'worried', 'anxious', 'terrified'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'stunned'],
            'trust': ['trust', 'reliable', 'dependable', 'honest', 'faithful'],
            'anticipation': ['expect', 'anticipate', 'await', 'looking forward']
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
        """Analyzes emotional tone of the text using spaCy."""
        doc = nlp(text)
        emotions = defaultdict(int)
        
        # Process each token
        for token in doc:
            lemma = token.lemma_.lower()
            for emotion, words in self.emotion_words.items():
                if lemma in words:
                    emotions[emotion] += 1
                    
            # Check for phrases (bigrams)
            if token.i < len(doc) - 1:
                bigram = f"{lemma} {doc[token.i + 1].lemma_.lower()}"
                for emotion, words in self.emotion_words.items():
                    if bigram in words:
                        emotions[emotion] += 1
        
        return dict(emotions) 
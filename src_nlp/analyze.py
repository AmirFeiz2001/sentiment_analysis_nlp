from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
import nltk

nltk.download('vader_lexicon', quiet=True)

def load_emotions(file_path):
    emotions = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
                if clear_line:
                    word, emotion = clear_line.split(':')
                    emotions[word] = emotion
        return emotions
    except Exception as e:
        print(f"Error loading emotions file: {e}")
        return {}

def analyze_emotions(words, emotions_dict):
    emotion_list = [emotions_dict[word] for word in words if word in emotions_dict]
    return Counter(emotion_list)

def sentiment_analyze(text):
    score = SentimentIntensityAnalyzer().polarity_scores(text)
    neg, pos = score['neg'], score['pos']
    if neg > pos:
        return 'Negative Sentiment'
    elif pos > neg:
        return 'Positive Sentiment'
    else:
        return 'Neutral Vibe'

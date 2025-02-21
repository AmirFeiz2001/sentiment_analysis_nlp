import string
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt', quiet=True)

def preprocess_text(text, stop_words=None):
    # Convert to lowercase
    lower_case = text.lower()
    
    # Remove punctuation
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokenized_words = word_tokenize(cleaned_text, "english")
    
    # Remove stop words if provided
    if stop_words:
        return [word for word in tokenized_words if word not in stop_words]
    return tokenized_words

def load_stop_words():
    return [
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
        "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
        "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
        "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
        "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
        "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
        "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
    ]

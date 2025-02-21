import numpy as np

def predict_sentiment(model, padded_text):
    prediction = model.predict(padded_text)[0][0]
    return 'positive sentiment' if prediction > 0.5 else 'negative sentiment'

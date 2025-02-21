from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.optimizers import Adam

def build_model(num_words, max_length, embedding_dim=32):
    model = Sequential([
        Embedding(num_words, embedding_dim, input_length=max_length),
        LSTM(64, dropout=0.1),
        Dense(1, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=3e-8)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

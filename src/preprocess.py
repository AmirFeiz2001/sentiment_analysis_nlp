from hazm import Normalizer, word_tokenize
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
import os

def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return pd.read_csv(f, delimiter=',', encoding='utf-8')
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_dataframe(df):

    train = df.rename(columns={df.columns[0]: 'text'})
    # لیبل گذاری به صورت دستی
    labels = ['0','1','0','1','0','1','1','0','1','1','0','0','0','1','1','1','1','0','1','0',
              '1','1','1','1','1','1','1','1','1','1','0','1','1','1','0','1','0','1','1','1',
              '1','0','1','0','0','1','1','0','1','0','0','1','0','0','1','1','1','1','1','0',
              '0','1','1','1','1','1','1','1','1','1','1','1','1','0','0','1','1','1','1','0',
              '0','1','1','0','1','1','1','1','1','1','1','1','1','0','1','0','1','1','0','1','1']
    train['target'] = [int(label) for label in labels[:len(train)]]
    return train

def counter_word(text):
    count = Counter()
    for sentence in text:
        for word in word_tokenize(sentence):
            count[word] += 1
    return count

def prepare_data(df, train_size_ratio=0.8, max_length=20):
    text = df['text']
    counter = counter_word(text)
    num_words = len(counter)

    train_size = int(len(df) * train_size_ratio)
    train_sentences = df['text'][:train_size]
    train_labels = df['target'][:train_size]
    test_sentences = df['text'][train_size:]
    test_labels = df['target'][train_size:]

    # Tokenize
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(train_sentences)
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    test_sequences = tokenizer.texts_to_sequences(test_sentences)

    # Pad sequences
    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
    test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

    return train_padded, test_padded, train_labels, test_labels, tokenizer, num_words

def preprocess_user_input(text_list, tokenizer, max_length=20):
    normalizer = Normalizer()
    normalized_texts = [normalizer.normalize(text) for text in text_list]
    sequences = tokenizer.texts_to_sequences(normalized_texts)
    return pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

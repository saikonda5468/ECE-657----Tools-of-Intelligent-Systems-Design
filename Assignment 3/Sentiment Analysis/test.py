import pandas as pd
import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences 
# Importing Libraries
from tensorflow.keras.utils import get_file
import tarfile
from glob import glob
import os,re,string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords,wordnet
import contractions
# Download the stopwords dataset from NLTK
nltk.download('stopwords')
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout,BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import LSTM, Bidirectional, GRU,Reshape
import joblib


max_sequence_length = 1000
# Load the tokenizer
tokenizer_save_path = 'Models/21036442_tokenizer.pkl'
tokenizer = joblib.load(tokenizer_save_path)

# Load the model
model_save_path = 'Models/21036442_NLP.h5'
loaded_model = load_model(model_save_path)


def expand_contractions(text):
    expanded_text = contractions.fix(text)
    return expanded_text
def remove_special_characters(text):
    special_chars = r'!"\'#$%&()*+/:;,<=>@[\\]^`{|}~'
    cleaned_text = re.sub('[' + re.escape(special_chars) + ']', '', text.lower())
    return cleaned_text
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Initialize lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Function to perform lemmatization and stemming
def lemmatize_and_stem(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Lemmatize each word based on its part of speech
    
    # Stem each word
    stem_words = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(stem_words)
import os
import pandas as pd

def read_imdb_data(directory_path):
    data = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read().strip()
            data.append(text)
    return data

def create_imdb_dataframe(negative_directory_path, positive_directory_path):
    negative_data = read_imdb_data(negative_directory_path)
    positive_data = read_imdb_data(positive_directory_path)

    # Create a pandas DataFrame with two columns: 'text' and 'sentiment'
    df = pd.DataFrame({'text': negative_data + positive_data,
                       'sentiment': [0] * len(negative_data) + [1] * len(positive_data)})
    
    return df


test_negative_directory_path = 'data/aclImdb/test/neg'
test_positive_directory_path = 'data/aclImdb/test/pos'


imdb_df_test = create_imdb_dataframe(test_negative_directory_path, test_positive_directory_path)



imdb_df_test['text'] = imdb_df_test['text'].apply(remove_special_characters)
imdb_df_test['text'] = imdb_df_test['text'].apply(remove_stopwords)
imdb_df_test['text'] = imdb_df_test['text'].apply(expand_contractions)
imdb_df_test['text'] = imdb_df_test['text'].apply(lemmatize_and_stem)



sequences_train = tokenizer.texts_to_sequences(imdb_df_test['text'])
padded_sequences_test = pad_sequences(sequences_train, maxlen=max_sequence_length,padding='post')



test_labels = imdb_df_test['sentiment'].values

scores = loaded_model.evaluate(padded_sequences_test,test_labels)
test_accuracy = scores[1]
print('accuracy on testing set:',test_accuracy*100)
     

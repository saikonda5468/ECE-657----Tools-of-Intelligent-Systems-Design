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

# Replace 'negative_directory_path' and 'positive_directory_path' with the actual paths to your IMDb dataset text files
train_negative_directory_path = 'data/aclImdb/train/neg'
train_positive_directory_path = 'data/aclImdb/train/pos'

# Load the IMDb dataset into a pandas DataFrame
imdb_df_train = create_imdb_dataframe(train_negative_directory_path, train_positive_directory_path)


# Now, you can work with the 'imdb_df' DataFrame, which contains the IMDb dataset
imdb_df_train['text'] = imdb_df_train['text'].apply(remove_special_characters)
imdb_df_train['text'] = imdb_df_train['text'].apply(remove_stopwords)
imdb_df_train['text'] = imdb_df_train['text'].apply(expand_contractions)
imdb_df_train['text'] = imdb_df_train['text'].apply(lemmatize_and_stem)



# Tokenization using Keras
max_num_words = 10000
max_sequence_length = 1000

tokenizer = Tokenizer(num_words=max_num_words)
tokenizer.fit_on_texts(imdb_df_train['text'])

sequences_train = tokenizer.texts_to_sequences(imdb_df_train['text'])
padded_sequences_train = pad_sequences(sequences_train, maxlen=max_sequence_length,padding='post')

labels = imdb_df_train['sentiment'].values
x_train, x_val, y_train, y_val = train_test_split(padded_sequences_train, labels, test_size=0.2, random_state=42)



embedding_dim = 100
filters1 = 128
filters2 = 64
kernel_size = 5
hidden_dims = 64
l2_lambda = 0.01

word_size = len(tokenizer.word_index)+1

model = keras.Sequential()
model.add(keras.layers.Embedding(word_size, 16))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Conv1D(filters=16,kernel_size=2,padding='valid',activation='relu'))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
history = model.fit(x_train,y_train,epochs=20,validation_data=(x_val, y_val),verbose=1,batch_size=512)

model_history = pd.DataFrame(history.history)

import matplotlib.pyplot as plt

# Access training and validation history
training_loss = model_history['loss']
validation_loss = model_history['val_loss']
training_accuracy = model_history['acc']
validation_accuracy = model_history['val_acc']

# # Plot training and validation loss
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.plot(range(0, 20), training_loss, label='Training Loss')
# plt.plot(range(0, 20), validation_loss, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Training and Validation Loss')

# # Plot training and validation accuracy
# plt.subplot(1, 2, 2)
# plt.plot(range(0, 20), training_accuracy, label='Training Accuracy')
# plt.plot(range(0, 20), validation_accuracy, label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.title('Training and Validation Accuracy')

# plt.tight_layout()  # Ensures proper spacing between subplots
# plt.show()


# Save the tokenizer
tokenizer_save_path = 'Models/21036442_tokenizer.pkl'
joblib.dump(tokenizer, tokenizer_save_path)

# Save the model
model_save_path = 'Models/21036442_NLP.h5'
model.save(model_save_path)

print("Tokenizer and Model saved successfully.")

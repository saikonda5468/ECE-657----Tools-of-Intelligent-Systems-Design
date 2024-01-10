# Importing Libraries
import numpy as np # LIBRARY IMPORT FOR LINEAR ALGEBRA
import pandas as pd # LIBRARY IMPORT FOR DATA PROCESSING
from sklearn.model_selection import train_test_split # MODULE IMPORT FOR DATA SPLITTING
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential
from tensorflow.keras.layers import Dense,LSTM, Dropout

from keras.models import load_model

# Load the model
model_save_path = 'Models/21036442_RNN.h5'
loaded_model = load_model(model_save_path)

print("Model loaded successfully.")

Test_data = pd.read_csv("Q2/test_data_RNN.csv")

y_test = Test_data['Target']
X_test = Test_data.drop(['Target'],axis=1)

# Convert Test_data to numpy arrays
X_test = np.array(X_test)
y_test = np.array(y_test)

# Reshape X_test to match the input shape of the LSTM model
X_test = np.expand_dims(X_test, axis=2)

# Make predictions using the loaded model
# predictions = loaded_model.predict(X_test)
# Make predictions using the loaded model
predictions = loaded_model.predict(X_test)

plt.figure(figsize=(20,10))
plt.plot(y_test, color="red", marker='o', linestyle='dashed', label="real stock price")
plt.plot(predictions, color="blue", marker='o', linestyle='dashed', label="predicted stock price")
plt.title("stock price prediction")
plt.xlabel("Date(random)")
plt.ylabel("stock price")
plt.legend()
plt.show()


scores=loaded_model.evaluate(X_test,y_test)
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

Stock_Data = pd.read_csv("./data/q2_dataset.csv")
store_data=np.zeros((1258,13))

for i in range(3, len(Stock_Data)):
        # Open prices
    store_data[i-3, 0] = Stock_Data.iloc[i-3, 3]  # open -1
    store_data[i-3, 1] = Stock_Data.iloc[i-2, 3]  # open -2
    store_data[i-3, 2] = Stock_Data.iloc[i-1, 3]  # open -3

        # High prices
    store_data[i-3, 3] = Stock_Data.iloc[i-3, 4]  # high -1
    store_data[i-3, 4] = Stock_Data.iloc[i-2, 4]  # high -2
    store_data[i-3, 5] = Stock_Data.iloc[i-1, 4]  # high -3

        # Low prices
    store_data[i-3, 6] = Stock_Data.iloc[i-3, 5]  # low -1
    store_data[i-3, 7] = Stock_Data.iloc[i-2, 5]  # low -2
    store_data[i-3, 8] = Stock_Data.iloc[i-1, 5]  # low -3

        # Volume
    store_data[i-3, 9] = Stock_Data.iloc[i-3, 2]  # volume -1
    store_data[i-3, 10] = Stock_Data.iloc[i-2, 2]  # volume -2
    store_data[i-3, 11] = Stock_Data.iloc[i-1, 2]  # volume -3

        # Target (next day's opening price)
    store_data[i-3, 12] = Stock_Data.iloc[i, 3]  # targe
    
col_names=['Open-1','Open-2','Open-3','High-1','High-2','High-3','Low-1','Low-2','Low-3','Volume-1','Volume-2','Volume-3','Target']
# Assuming you have a numpy array named 'data'
data = np.array(store_data[:-2, :])  # Replace the ellipsis with your actual data

# Shuffle the array
np.random.shuffle(data)

df = pd.DataFrame(data, columns=col_names)
print(df)
df_new=df.drop(['Target'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(df_new, df['Target'], test_size=0.3, random_state = 42) 

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Data Split and Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert NumPy arrays to Pandas DataFrames
X_train_df = pd.DataFrame(X_train, columns=df_new.columns)
X_test_df = pd.DataFrame(X_test, columns=df_new.columns)
y_train_df = pd.DataFrame(y_train, columns=['Target'])
y_test_df = pd.DataFrame(y_test, columns=['Target'])

# Concatenate X_test_df and y_test_df
test_data = pd.concat([X_test_df, y_test_df], axis=1)

# Concatenate X_train_df and y_train_df
train_data = pd.concat([X_train_df, y_train_df], axis=1)

# Save to CSV files
train_data.to_csv("Q2/train_data_RNN.csv", index=False, header=True)
test_data.to_csv("Q2/test_data_RNN.csv", index=False, header=True)

label = train_data['Target']
train_data=train_data.drop(['Target'],axis=1)


train_data = np.expand_dims(train_data, axis=2)

X_train, X_val, y_train, y_val = train_test_split(train_data, label, test_size=0.2, random_state=42)



    # 2. Train your network
model = Sequential()
    #adding LSTM layer with 50 LSTM units
model.add(LSTM(50,input_shape=(X_train.shape[1],1),return_sequences=True))
model.add(Dropout(0.2)) 
model.add(LSTM(150))
model.add(Dropout(0.2)) 
    #adding dense layer
model.add(Dense(1,activation='linear'))

    #'mean_absolute_error' has been used as loss function

model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['mae'])

History = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=600,batch_size=64,verbose=1)

model.save("Models/21036442_RNN.h5")

# Print model summary
model.summary()
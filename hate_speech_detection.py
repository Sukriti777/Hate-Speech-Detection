import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and preprocess the data
data = pd.read_csv('labeled_data.csv')
X = data['tweet']
y = data['class']  # Assuming 'class' column contains the labels

# Convert labels to binary (0: hate speech, 1: offensive language, 2: neither)
y = y.apply(lambda x: 1 if x == 0 else 0)  # Modify as needed

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)
print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr))

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train_tfidf, y_train)
y_pred_rf = rf_model.predict(X_test_tfidf)
print("Random Forest Report:")
print(classification_report(y_test, y_pred_rf))

# Prepare data for LSTM and CNN
max_words = 5000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# LSTM Model
lstm_model = Sequential()
lstm_model.add(Embedding(max_words, 128, input_length=max_len))
lstm_model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.2)
y_pred_lstm = (lstm_model.predict(X_test_pad) > 0.5).astype("int32")
print("LSTM Model Report:")
print(classification_report(y_test, y_pred_lstm))

# CNN Model
cnn_model = Sequential()
cnn_model.add(Embedding(max_words, 128, input_length=max_len))
cnn_model.add(Conv1D(128, 5, activation='relu'))
cnn_model.add(MaxPooling1D(pool_size=4))
cnn_model.add(GlobalMaxPooling1D())
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dense(1, activation='sigmoid'))

cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn_model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.2)
y_pred_cnn = (cnn_model.predict(X_test_pad) > 0.5).astype("int32")
print("CNN Model Report:")
print(classification_report(y_test, y_pred_cnn))

# Hybrid Model (Example: Averaging predictions)
y_pred_rf = y_pred_rf.reshape(-1, 1)
y_pred_lstm = y_pred_lstm.reshape(-1, 1)
y_pred_cnn = y_pred_cnn.reshape(-1, 1)

y_pred_hybrid = (y_pred_rf + y_pred_lstm + y_pred_cnn) / 3
y_pred_hybrid = (y_pred_hybrid > 0.5).astype(int).reshape(-1)

print("Hybrid Model Report:")
print(classification_report(y_test, y_pred_hybrid))

# Function to preprocess and predict sample text
def predict_sample_text(text):
    # Preprocess the sample text
    text_tfidf = vectorizer.transform([text])
    text_seq = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_seq, maxlen=max_len)

    # Get predictions from each model
    pred_lr = lr_model.predict(text_tfidf)
    pred_rf = rf_model.predict(text_tfidf).reshape(-1, 1)
    pred_lstm = (lstm_model.predict(text_pad) > 0.5).astype("int32").reshape(-1, 1)
    pred_cnn = (cnn_model.predict(text_pad) > 0.5).astype("int32").reshape(-1, 1)

    # Combine predictions for the hybrid model
    pred_hybrid = (pred_rf + pred_lstm + pred_cnn) / 3
    pred_hybrid = (pred_hybrid > 0.5).astype(int).reshape(-1)

    # Output the prediction
    if pred_hybrid[0] == 1:
        print("The text is classified as hate speech.")
    else:
        print("The text is not classified as hate speech.")

# Main loop to get user input and predict
while True:
    sample_text = input("Enter a text to check for hate speech (or type 'exit' to quit): ")
    if sample_text.lower() == 'exit':
        break
    predict_sample_text(sample_text)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras 
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam

# Load the dataset
train_data = pd.read_csv('train.csv')
train_data = train_data.fillna('')
train_texts = train_data['text'].values
train_labels = train_data['label'].values

# Preprocess the data
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
train_data = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=500)

# Define the DANN model
def build_model():
    input_layer = Input(shape=(500,))
    x = Dense(128, activation='relu')(input_layer)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output_layer = Dense(20, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

dann_model = build_model()
dann_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Train the DANN model
dann_model.fit(train_data, train_labels, epochs=5, batch_size=32)

def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=500)
    return padded_sequence

dann_model.save('dann_model.h5')
# Choose a random instance to predict
idx = np.random.randint(0, len(train_data))
text_instance = train_texts[idx]
true_label = train_labels[idx]
print('Text instance: \n', text_instance)
text_instance = preprocess_text(text_instance)

# Predict the label using the DANN model
predicted_label = dann_model.predict(text_instance)
predicted_label = np.argmax(predicted_label, axis=-1)
print('Predicted label: ', predicted_label)
print('True label: ', true_label)

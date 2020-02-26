import warnings
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import gc


vocab_size = 10000
embedding_dim = 64
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
max_length = 200

warnings.filterwarnings("ignore")
warnings.filterwarnings("error", message=".*check_inverse*.", category=UserWarning, append=False)

dataAll = pd.read_csv("Train.csv")
dataAll, data = train_test_split(dataAll, test_size=0.1, random_state=1)
del dataAll

data, test = train_test_split(data, test_size=0.2, random_state=1)

gc.collect()

questionTrain = data['Title']+data['Body']
tagsTrain = data['Tags']

questionTest = test['Title']+test['Body']
tagsTest = test['Tags']


tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(data)

word_index = tokenizer.word_index

del data
del test

questionTrain_sequences = tokenizer.texts_to_sequences(questionTrain)
del questionTrain
questionTrain_padded = pad_sequences(questionTrain_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

tagsTrain=tagsTrain.astype('str')
tagsTrain_sequences = tokenizer.texts_to_sequences(tagsTrain)
del tagsTrain
tagsTrain_padded = pad_sequences(tagsTrain_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

questionTest_sequences = tokenizer.texts_to_sequences(questionTest)
del questionTest
questionTest_padded = pad_sequences(questionTest_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

tagsTest=tagsTest.astype('str')
tagsTest_sequences = tokenizer.texts_to_sequences(tagsTest)
del tagsTest
tagsTest_padded = pad_sequences(tagsTest_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)



BUFFER_SIZE = 10000
BATCH_SIZE = 64
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(150, activation='sigmoid')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

NUM_EPOCHS = 5
history = model.fit(questionTrain_padded, tagsTrain_padded, epochs=NUM_EPOCHS, validation_data=(questionTest_padded, tagsTest_padded), verbose=1)

results = model.evaluate(questionTest_padded, tagsTest_padded, batch_size=BATCH_SIZE)

gc.collect()

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')














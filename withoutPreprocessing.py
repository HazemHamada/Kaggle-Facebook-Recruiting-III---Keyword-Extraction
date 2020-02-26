import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import lightgbm as lgb
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer
import nltk
from nltk.tokenize import word_tokenize
import gc
import re
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,precision_score,recall_score
import warnings

vocab_size = 10000
embedding_dim = 32
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

warnings.filterwarnings("ignore")
warnings.filterwarnings("error", message=".*check_inverse*.", category=UserWarning, append=False)

dataAll = pd.read_csv("Train.csv")
dataAll, data = train_test_split(dataAll, test_size=0.2, random_state=1)

data, test = train_test_split(dataAll, test_size=0.2, random_state=1)
del dataAll


question = data['Title']+data['Body']
tags=data['Tags']
data=tags.join(question)

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(data)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(data)
training_padded = pad_sequences(training_sequences, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(test)
testing_padded = pad_sequences(testing_sequences, padding=padding_type, truncating=trunc_type)

BUFFER_SIZE = 10000
BATCH_SIZE = 64
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

NUM_EPOCHS = 10
history = model.fit(data, epochs=NUM_EPOCHS, validation_data=test)

import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')














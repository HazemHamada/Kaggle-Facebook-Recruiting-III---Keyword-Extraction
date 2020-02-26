import re
import warnings
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm

vocab_size = 10000
embedding_dim = 32
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
max_length = 200

warnings.filterwarnings("ignore")
warnings.filterwarnings("error", message=".*check_inverse*.", category=UserWarning, append=False)

dataAll = pd.read_csv("Train.csv")
dataAll, data = train_test_split(dataAll, test_size=0.2, random_state=1)

#data, test = train_test_split(data, test_size=0.2, random_state=1)
del dataAll


# preprocessing

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
qus_list=[]
qus_with_code = 0
len_before_preprocessing = 0
len_after_preprocessing = 0
for index,row in data.iterrows():
    title, body, tags = row["Title"], row["Body"], row["Tags"]
    if '<code>' in body:
        qus_with_code+=1
    len_before_preprocessing+=len(title) + len(body)
    body=re.sub('<code>(.*?)</code>', '', body, flags=re.MULTILINE|re.DOTALL)
    body = re.sub('<.*?>', ' ', str(body.encode('utf-8')))
    title=title.encode('utf-8')
    question=str(title)+" "+str(body)
    question=re.sub(r'[^A-Za-z]+',' ',question)
    words=word_tokenize(str(question.lower()))
    question=' '.join(str(stemmer.stem(j)) for j in words if j not in stop_words and (len(j)!=1 or j=='c'))
    qus_list.append(question)
    len_after_preprocessing += len(question)
data["question"] = qus_list
avg_len_before_preprocessing=(len_before_preprocessing*1.0)/data.shape[0]
avg_len_after_preprocessing=(len_after_preprocessing*1.0)/data.shape[0]
print( "Avg. length of questions(Title+Body) before preprocessing: ", avg_len_before_preprocessing)
print( "Avg. length of questions(Title+Body) after preprocessing: ", avg_len_after_preprocessing)
print ("% of questions containing code: ", (qus_with_code*100.0)/data.shape[0])
data = data[["question","Tags"]]
print("Shape of preprocessed data :", data.shape)


######################################################################
gc.collect()

data, test = train_test_split(data, test_size=0.2, random_state=1)
questionTrain = data['question']
tagsTrain = data['Tags']
questionTest = test['question']
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
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(150, activation='sigmoid')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

NUM_EPOCHS = 10
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

















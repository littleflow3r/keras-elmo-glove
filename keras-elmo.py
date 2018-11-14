import tensorflow as tf
  2 import pandas as pd
  3 import tensorflow_hub
  4 import os, re
  5 import keras.layers as layers
  6 from keras.models import Model, Sequential
  7 from keras.layers import Lambda, Dense, Input
  8 import numpy as np
  9
 10 # prepare the data
 11 def load_dataset(dir):
 12     dirall = [dir+"/pos", dir+"/neg"]
 13     for d in dirall:
 14         data = {}
 15         data["text"] = []
 16         data["score"] = []
 17         for files in os.listdir(d):
 18             with tf.gfile.GFile(d+"/"+files, "r") as f:
 19                 score = re.match("\d+_(\d+)\.txt", files).group(1)
 20                 data["text"].append(f.read())
 21                 data["score"].append(score)
 22
 23         if d[-3:] == 'pos':
 24             pos = pd.DataFrame.from_dict(data)
 25             pos["sentiment"] = 1
 26         elif d[-3:] == 'neg':
 27             neg = pd.DataFrame.from_dict(data)
 28             neg["sentiment"] = 0
 29     return pd.concat([pos, neg]).sample(frac=1).reset_index(drop=True)
 30
 31 train = load_dataset("aclImdb/train")
 32 test = load_dataset("aclImdb/test")
 33 sample = 200
 34 maxlen = 150
 35 x_train = train['text'][:sample].tolist()
 36 x_train = [' '.join(t.split()[0:maxlen]) for t in x_train]
 37 x_train = np.array(x_train, dtype=object)[:, np.newaxis]
 38 y_train = train['sentiment'][:sample].tolist()
 39
 40 x_test = test['text'][:sample].tolist()
 41 x_test = [' '.join(t.split()[0:maxlen]) for t in x_test]
 42 x_test = np.array(x_test, dtype=object)[:, np.newaxis]
 43 y_test = test['sentiment'][:sample].tolist()
 44
 45
 46 print ("loading elmo...:")
 47 elmo = tensorflow_hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
 48 #casting the input as string
 49 def elmo_embed(x):
 50     return elmo(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=    True)["default"]
 52 #defining the model architecture
 53 def build_model():
 54     input_t = layers.Input(shape=(1,), dtype=tf.string)
 55     elmo_layer = layers.Lambda(elmo_embed)(input_t)
 56     dense1 = layers.Dense(32, activation='relu')(elmo_layer)
 57     dense2 = layers.Dense(1, activation='sigmoid')(dense1)
 58     return Model(inputs=[input_t],outputs=dense2)
 59
 60
 61 model = build_model()
 62 print (model.summary())
 63
 64 model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
 65
 66 print ('fit........')
 67 history = model.fit(x_train, y_train,
 68         epochs=10,
 69         batch_size=10,
 70         validation_data=(x_test, y_test))
 71 print ('history:', history.history)
 72
 73 import matplotlib.pyplot as plt
 74
 75 val_acc = history.history['val_acc']
 76 val_loss = history.history['val_loss']
 77
 78 epochs = range(1, len(val_acc) +1)
 79
 80 plt.plot(epochs, val_loss, 'bo', label='Test loss')
 81 plt.plot(epochs, val_acc, 'b', label='Test acc')
 82 plt.title('Test loss and acc')
 83 plt.legend()
 84 plt.savefig('test.png')


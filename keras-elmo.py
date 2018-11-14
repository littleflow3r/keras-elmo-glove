import tensorflow as tf
import pandas as pd
import tensorflow_hub
import os, re
import keras.layers as layers
from keras.models import Model, Sequential
from keras.layers import Lambda, Dense, Input
import numpy as np

# prepare the data
def load_dataset(dir):
        dirall = [dir+"/pos", dir+"/neg"]
        for d in dirall:
                data = {}
                data["text"] = []
                data["score"] = []
                for files in os.listdir(d):
                        with tf.gfile.GFile(d+"/"+files, "r") as f:
                                score = re.match("\d+_(\d+)\.txt", files).group(1)
                                data["text"].append(f.read())
                                data["score"].append(score)

                if d[-3:] == 'pos':
                        pos = pd.DataFrame.from_dict(data)
                        pos["sentiment"] = 1
                elif d[-3:] == 'neg':
                        neg = pd.DataFrame.from_dict(data)
                        neg["sentiment"] = 0
        return pd.concat([pos, neg]).sample(frac=1).reset_index(drop=True)

train = load_dataset("aclImdb/train")
test = load_dataset("aclImdb/test")
sample = 200
maxlen = 150
x_train = train['text'][:sample].tolist()
x_train = [' '.join(t.split()[0:maxlen]) for t in x_train]
x_train = np.array(x_train, dtype=object)[:, np.newaxis]
y_train = train['sentiment'][:sample].tolist()

x_test = test['text'][:sample].tolist()
x_test = [' '.join(t.split()[0:maxlen]) for t in x_test]
x_test = np.array(x_test, dtype=object)[:, np.newaxis]
y_test = test['sentiment'][:sample].tolist()


print ("loading elmo...:")
elmo = tensorflow_hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
#casting the input as string
def elmo_embed(x):
        return elmo(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

#defining the model architecture
def build_model():
        input_t = layers.Input(shape=(1,), dtype=tf.string)
        elmo_layer = layers.Lambda(elmo_embed)(input_t)
        dense1 = layers.Dense(32, activation='relu')(elmo_layer)
        dense2 = layers.Dense(1, activation='sigmoid')(dense1)
        return Model(inputs=[input_t],outputs=dense2)


model = build_model()
print (model.summary())

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

print ('fit........')
history = model.fit(x_train, y_train,
                epochs=10,
                batch_size=10,
                validation_data=(x_test, y_test))
print ('history:', history.history)

import matplotlib.pyplot as plt

val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

epochs = range(1, len(val_acc) +1)

plt.plot(epochs, val_loss, 'bo', label='Test loss')
plt.plot(epochs, val_acc, 'b', label='Test acc')
plt.title('Test loss and acc')
plt.legend()
plt.savefig('test.png')

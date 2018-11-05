import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os, re
from keras import backend as K
import keras.layers as layers
from keras.models import Model
import numpy as np

sess = tf.Session()
K.set_session(sess)

def load_directory_data(directory):
	data = {}
	data["sentence"] = []
	data["sentiment"] = []
	for file_path in os.listdir(directory):
		with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
			data["sentence"].append(f.read())
			data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))

	return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
	pos_df = load_directory_data(os.path.join(directory, "pos"))
	neg_df = load_directory_data(os.path.join(directory, "neg"))
	pos_df["polarity"] = 1
	neg_df["polarity"] = 0
	return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


tf.logging.set_verbosity(tf.logging.ERROR)
dataset = "aclImdb"
train_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "train"))

test_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "test"))
print (train_df.head())

print ('elmo model.....')
elmo_model = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

def ElmoEmbedding(x):
	return elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

input_text = layers.Input(shape=(1,), dtype=tf.string) 
embedding = layers.Lambda(ElmoEmbedding, output_shape=(1024,))(input_text)
dense = layers.Dense(256, activation='relu')(embedding)
pred = layers.Dense(1, activation='sigmoid')(dense)

model = Model(inputs=[input_text],outputs=pred)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

train_text = train_df['sentence'][:200].tolist()
train_text = [' '.join(t.split()[0:150]) for t in train_text]
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
train_label = train_df['polarity'][:200].tolist()

test_text = test_df['sentence'][:100].tolist()
test_text = [' '.join(t.split()[0:150]) for t in test_text]
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
test_label = test_df['polarity'][:100].tolist()

print ('fit........')
history = model.fit(train_text, train_label, validation_data=(test_text, test_label), epochs=5, batch_size=32)
print (history.history)

import matplotlib.pyplot as plt

val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

epochs = range(1, len(val_acc) +1)

plt.plot(epochs, val_loss, 'bo', label='Test loss')
plt.plot(epochs, val_acc, 'b', label='Test acc')
plt.title('Test loss and acc')
plt.legend()
plt.savefig('test.png')

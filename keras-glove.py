#preprocess the labels of the raw data
import os
imdb_dir = 'aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
	dir_name = os.path.join(train_dir, label_type)
	for fname in os.listdir(dir_name):
		if fname[-4:] == '.txt':
			f = open(os.path.join(dir_name, fname))
			texts.append(f.read())
			f.close()
			if label_type == 'neg':
				labels.append(0)
			else:
				labels.append(1)


#tokenizing the data
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 150
training_samples = 200
validation_samples = 100
max_words = 10000 #consider only the top 10000 words in dataset

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print ('Found %s unique tokens.' %len(word_index))


data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print ('Shape of data tensor:', data.shape)
print ('Shape of label tensor:', labels.shape)

#splits the data into a training and validation set but first shuffles the data (because the data samples are ordered all negative first and then all positive)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

#parsing the Glove word-embeddings file
embeddings_index = {}
f = open('glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print ('Found %s word vectors.' %len(embeddings_index))

#preparing the Glove word-embeddings matrix
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
	if i < max_words:
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

#defining the model
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False #to freeze the layer, so that the pretrained parts are not updated during training
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print (model.summary())

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=10, validation_data=(x_val, y_val))
#model.save_weights('pretrained_glove_model.h5')

print (history.history)
#import sys
#sys.exit()
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) +1)


plt.plot(epochs, val_loss, 'bo', label='Test loss')
plt.plot(epochs, val_acc, 'b', label='Test acc')
plt.title('Test loss and acc')
plt.legend()
plt.savefig('test.png')

import sys
sys.exit()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.legend()
plt.savefig('train-val-acc.png')

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('train-val-loss.png')
plt.show()

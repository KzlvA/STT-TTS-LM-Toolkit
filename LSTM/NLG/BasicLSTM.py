# Load LSTM network and generate text
# result.txt is a file composed of subtitles from all collected vtt files
import sys
import numpy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import utils

# load ascii text and covert to lowercase
filename = "[absolute file path].txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = utils.to_categorical(dataY)
# define the LSTM model - single hidden LSTM layer with 256 memory units.
# The network uses dropout with a probability of 20. The output layer is
# a Dense layer using the softmax activation function to output a
# probability prediction for each of the 47 characters between 0 and 1.
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
# optimizing the log loss (cross entropy), here using the ADAM
# optimization algorithm for speed.
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# use model checkpointing to record all of the network weights to file each time
# an improvement in loss is observed at the end of the epoch. We will use the
# best set of weights (lowest loss) to instantiate our generative model
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model - modest number of 20 epochs and a large batch size of 128 patterns.
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)

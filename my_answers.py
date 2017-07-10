import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras

#Added by Brian
import string

#  Transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    step_count = len(series) - window_size
    for i in range(step_count):
        view = i+window_size
        X.append(series[i:view])
        y.append(series[view])


    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# Build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    
    model.add(LSTM(3, input_shape=(window_size, 1)))
    model.add(Dense(1, activation="tanh"))
    
    return model


### Return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    ascii_lowercase = list(string.ascii_lowercase)
    white_list = punctuation + ascii_lowercase + [' ']
    text = "".join(filter(white_list.__contains__, text))
    return text

### Function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    step_count = len(text)-window_size-step_size
    for i in range(0, step_count, step_size):
        view = i+window_size
        inputs.append(text[i:view])
        outputs.append(text[view])

    return inputs,outputs

# Build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation="linear"))
    model.add(Dense(num_chars, activation="softmax"))

    return model
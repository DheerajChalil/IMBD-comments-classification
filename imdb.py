#binary text classification
import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers,optimizers
import matplotlib as plt
#loading data from the dataset
(train_data,train_labels), (test_data,test_labels) = imdb.load_data(num_words = 10000)
#keeping only top 10000 repeated words in th etraining data and discarding the others
#creating binary matrix of the given data
def vector_sequences(sequences,dimension = 1000):
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1
    return results
#feeding training data in the x branch
x_train = vector_sequences(train_data)
x_test = vector_sequences(test_data)
#vectorizing and feeding label in the y branch
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('floate32')
#Adding model layers and definition
model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape = (10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
#making an optimizer and compiling
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
#testing model accuracy using validation
#slicing data
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
#sliciing labels
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
#Not sure how many epochs to give...currently giving 10
history = model.fit(partial_x_train,partial_y_train,epochs=10,batch_size=512,validation_data=(x_val,y_val))







import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import genfromtxt
from tensorflow import keras

raw_train = genfromtxt('./digits_train.csv', delimiter=',')
raw_test = genfromtxt('./digits_test.csv', delimiter=',')

raw_train = raw_train[1:]
np.random.shuffle(raw_train)


X_train = raw_train[0:29400]
X_test = raw_train[29401:]

y_train = X_train[:,0]
y_test = X_test[:,0]
X_train = X_train[:,1:]
X_test = X_test[:,1:]

train_images = []
test_images = []
for i in range(len(X_train)):
    train_images.append(X_train[i].reshape((28, 28)))

for i in range(len(X_test)):
    test_images.append(X_test[i].reshape((28, 28)))

X_train = X_train/255.0
X_test = X_test/255.0

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>=0.97):
            print("\nReached {}% accuracy so cancelling training!\n".format(logs.get('acc')*100))
            self.model.stop_training = True

callback = myCallback()
model = keras.models.Sequential()
model.add(keras.layers.Dense(784))
model.add(keras.layers.Dense(1024, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, callbacks=[callback])

model.evaluate(X_test, y_test)

predictions = model.fit(X_test)

labels = np.argmax(predictions, axis=1)








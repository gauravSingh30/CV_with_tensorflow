import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
import pandas as pd
plt.interactive(False)

print(tf.__version__)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') > 0.85):
            print("\naccuracy reached at {} hence stopping \n".format(logs.get('acc')*100))
            self.model.stop_training = True


raw_test = genfromtxt('./data/fashionmnist/fashion-mnist_test.csv', delimiter=',')
raw_train = genfromtxt('./data/fashionmnist/fashion-mnist_train.csv', delimiter=',')

test = raw_test[1:]
train = raw_train[1:]

np.random.shuffle(test)
np.random.shuffle(train)

X_train = train[:,1:]
y_train = train[:,0]
X_test = test[:,1:]
y_test = test[:,0]

train_images = []
test_images = []
for i in range(np.shape(train)[0]):
    train_images.append(X_train[i].reshape((28,28)))

for i in range(np.shape(test)[0]):
    test_images.append(X_test[i].reshape((28,28)))


plt.imshow(train_images[0])

X_train = X_train/255.0
X_test = X_test/255.0

callback = myCallback()
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(784))
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, callbacks=[callback])

model.evaluate(X_test, y_test)

classifications = model.predict(X_test)

labels = np.argmax(classifications, axis=1)

image_dict = {
    0 : 'T-shirt/top',
    1 : 'Trouser',
    2 : 'Pullover',
    3 : 'Dress',
    4 : 'Coat',
    5 : 'Sandal',
    6 : 'Shirt',
    7 : 'Sneaker',
    8 : 'Bag',
    9 : 'Ankle boot',
}

classes = []
for i in range(len(labels)):
    classes.append(image_dict[labels[i]])


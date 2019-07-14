# Single Neuron NN with logic that y = 50(x+1)

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 11.0, 15.0], dtype=float)
ys = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 550.0, 600.0, 800.0], dtype=float)

mean = np.mean(ys)
std = np.std(ys)
ys = (ys - mean)/std
model = tf.keras.Sequential()
model.add(layers.Dense(units=1, input_shape=[1]))
model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(xs, ys, epochs=1000)

out = model.predict([7.0])
original = out*std + mean
print("Non Scaled: {} and Scaled: {}".format(original, out))

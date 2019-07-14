import os
import zipfile
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

## Unzipping Files
localzip = './horse-or-human.zip'
validationzip = './validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(localzip, 'r')
zip_ref.extractall('./horse-or-human')
zip_ref.close()
zip_ref = zipfile.ZipFile(validationzip, 'r')
zip_ref.extractall('./validation-horse-or-human')
zip_ref.close()


## Directories for Image Flow
train_horse_dir = './horse-or-human/horses/'
train_human_dir = './horse-or-human/humans/'
validation_horse_dir = './validation-horse-or-human/horses/'
validation_human_dir = './validation-horse-or-human/humans/'

train_horses_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)
validation_horses_names = os.listdir(validation_horse_dir)
validation_humans_names = os.listdir(validation_human_dir)

print('total training horses: ', len(os.listdir(train_horse_dir)))
print('total training humans: ', len(os.listdir(train_human_dir)))
print('total validation horses: ', len(os.listdir(validation_horse_dir)))
print('total validation humans: ', len(os.listdir(validation_human_dir)))

# %matplotlib inline

## Plotting Images for sampling
n_rows = 4
n_cols = 4
pic_index = 0

fig = plt.gcf()
fig.set_size_inches(n_rows*4, n_cols*4)
pic_index = pic_index + 8

train_horses_pix = [os.path.join(train_horse_dir, fname)
                    for fname in train_horses_names[pic_index-8:pic_index]]

train_human_pix = [os.path.join(train_human_dir, fname)
                  for fname in train_human_names[pic_index-8:pic_index]]

for i, img in enumerate(train_horses_pix+train_human_pix):
    sp = plt.subplot(n_rows, n_cols, i+1)
    sp.axis('off')
    img_path = mpimg.imread(img)
    plt.imshow(img_path)

plt.show()


## Simple Model with 5 convolutions
model = keras.Sequential()
model.add(keras.layers.Conv2D(16, (3,3), activation=tf.nn.relu, input_shape=(300, 300, 3)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))


from tensorflow.keras.optimizers import RMSprop

model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001), metrics=['acc'])

##Training with Image Generator with validation images

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=(1/255))
train_generator = train_datagen.flow_from_directory(
                    './horse-or-human',
                    target_size=(300,300),
                    batch_size=128,
                    class_mode='binary')
validation_datagen = ImageDataGenerator(rescale=(1/255))
validation_generator = validation_datagen.flow_from_directory(
                        './validation-horse-or-human/',
                        target_size=(300,300),
                        batch_size=32,
                        class_mode='binary')

history = model.fit_generator(
          train_generator,
          steps_per_epoch=8,
          epochs=2,
          verbose=1,
          validation_data=validation_generator,
          validation_steps=8)

## Testing the model with samples
import numpy as np

from keras.preprocessing import image

img = image.load_img('./horse1.jpeg', target_size=(300, 300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
classes = model.predict(x)
if classes > 0.5:
    print("this is a human")
else:
    print("this is a horse")

plt.imshow(img)


## Building the convolution map for seeing the important features
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output for layer in model.layers[1:]]
#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
# Let's prepare a random input image from the training set.
horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horses_names]
human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
img_path = random.choice(horse_img_files + human_img_files)

img = load_img(img_path, target_size=(300, 300))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers]

# Now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    if len(feature_map.shape) == 4:
        # Just do this for the conv / maxpool layers, not the fully-connected layers
        n_features = feature_map.shape[-1]  # number of features in feature map
        # The feature map has shape (1, size, size, n_features)
        size = feature_map.shape[1]
        # We will tile our images in this matrix
        display_grid = np.zeros((size, size * n_features))
        for i in range(n_features):
          # Postprocess the feature to make it visually palatable
            x = feature_map[0, :, :, i]
            x -= x.mean()
            x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')
          # We'll tile each filter into this big horizontal grid
            display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid
        scale = 20. / n_features
        plt.figure(figsize=(scale * n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')



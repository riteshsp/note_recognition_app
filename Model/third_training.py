import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from keras._tf_keras.keras.models import Sequential, load_model, save_model
import keras._tf_keras.keras.layers as layers


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        rotation_range=30,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    "/home/user/Documents/learning/AI python/Tensorflow/archive/Indian currency dataset v1/training",
    target_size=(64,64),
    batch_size=32,
    shuffle=True,
    seed=42,
    color_mode="rgb",
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "/home/user/Documents/learning/AI python/Tensorflow/archive/Indian currency dataset v1/validation/",
    target_size=(64,64),
    batch_size=32,
    shuffle=True,
    seed=42,
    color_mode="rgb",
    class_mode='categorical')


model = Sequential([
      layers.Conv2D(128, (3,3), activation='relu', input_shape=(64, 64, 3)),
      layers.MaxPooling2D(2, 2),
      layers.Conv2D(64, (3,3), activation='relu'),
      layers.MaxPooling2D(2,2),
#       tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
#       tf.keras.layers.MaxPooling2D(2,2),
      layers.Conv2D(32, (3,3), activation='relu'),
      layers.MaxPooling2D(2,2),
      layers.Flatten(),
      layers.Dense(512, activation='relu'),
      layers.Dense(8, activation='softmax')
])

model.compile(optimizer = 'adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

print(model.summary())

epochs = 10

history = model.fit(x = train_generator,validation_data=test_generator ,batch_size=32,verbose=1, epochs=epochs)
print("saving model")
save_model(model, 'py11model2.h5')
print("model saved")
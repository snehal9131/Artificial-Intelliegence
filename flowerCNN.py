import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow_core.python.keras import Input, Model

import os
import numpy as np
import matplotlib.pyplot as plt

#number of samples propagated through the network per iteration
from tensorflow_core.python.keras.layers import Dropout

batch_size = 128
#number of epochs: pass the entire dataset through the netowork
epochs = 15
#Preset the img height and width
IMG_HEIGHT = 64
IMG_WIDTH = 64

#train_image_generator = ImageDataGenerator(rescale=1./255)
#validation_image_generator = ImageDataGenerator(rescale=1./255)
#test_image_generator = ImageDataGenerator(rescale=1./255)

#Train Test Split with the test folder (validation split)
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(directory='data/train',
                                                    target_size=(IMG_HEIGHT,IMG_WIDTH),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset = 'training')
validation_generator = train_datagen.flow_from_directory(directory='data/train',
                                                    target_size=(IMG_HEIGHT,IMG_WIDTH),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset = 'validation')

# train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
#                                                            directory='data/train',
#                                                            shuffle=True,
#                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
#                                                            class_mode='categorical')
# val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
#                                                               directory='data/validation',
#                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
#                                                               class_mode='categorical')
# test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
#                                                               directory='data/test',
#                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
#                                                               class_mode='categorical')

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    BatchNormalization(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(500, activation='relu'),
    Dense(300, activation='relu'),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
               metrics=['accuracy'])
model.summary()
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data= validation_generator
)
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
test_loss, test_acc = model.evaluate(train_generator, verbose=2)
print(test_acc)

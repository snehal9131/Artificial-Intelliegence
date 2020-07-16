from utils import load_data

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split

#load the data from the utils file
(feature, labels) = load_data()

#perform a train/test split for the features and lables of our dataset
x_train, x_test,y_train,y_test = train_test_split(feature,labels,test_size=0.1)

#define the categories
categories = ['daisy', 'sunflower','rose', 'tulip','dandelion']

#define the Input shape of the CNN
input_layer = tf.keras.layers.Input([224,224,3])

#define the model with 4 Conv2d layers with maxpooling, 2 Dense Layers (ending with 5)
conv1 = tf.keras.layers.Conv2D(filters = 32, kernel_size=(5,5), padding='Same',activation='relu')(input_layer)
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)

conv2 = tf.keras.layers.Conv2D(filters = 64, kernel_size=(3,3), padding='Same',activation='relu')(pool1)
pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv2)

conv3 = tf.keras.layers.Conv2D(filters = 96, kernel_size=(3,3), padding='Same',activation='relu')(pool2)
pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv3)

conv4 = tf.keras.layers.Conv2D(filters = 96, kernel_size=(3,3), padding='Same',activation='relu')(pool3)
pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv4)

flt1 = tf.keras.layers.Flatten()(pool4)

dn1 = tf.keras.layers.Dense(512, activation='relu')(flt1)
out = tf.keras.layers.Dense(5, activation='softmax')(dn1)

model = tf.keras.Model(input_layer, out)

validation_data=(x_test,y_test)
#compile the model
model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#fit the model with and define the batchs size + number of epochs
history = model.fit(x_train,y_train, batch_size= 200, epochs = 30, validation_data=validation_data)
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
#save the model
model.save('mymodel.h5')




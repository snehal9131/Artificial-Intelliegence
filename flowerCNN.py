import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

batch_size = 256
#number of epochs: pass the entire dataset through the netowork
epochs = 100
#Preset the img height and width
IMG_HEIGHT = 64
IMG_WIDTH = 64

#Train Test Split with the test folder (validation split)
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range= 20, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(directory='data/train',
                                                    target_size=(IMG_HEIGHT,IMG_WIDTH),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset = 'training')
validation_generator = train_datagen.flow_from_directory(directory='data/train',
                                                    target_size=(IMG_HEIGHT,IMG_WIDTH),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset = 'validation',
                                                    shuffle= False)

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
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
    validation_data=validation_generator
)
model.save('cnn.h5')

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

X = validation_generator.next()
Xarr = np.array(X[0])
for i in range(5):
    X = validation_generator.next()
    np.append(Xarr,(X[0]))
print (Xarr)
Yarr = np.array(validation_generator.labels)
#for i in range(len(X)):
print(Xarr[0].shape)
print (Yarr)
plt.clf()

img_nr = 3
plt.imshow(Xarr[img_nr])
Ypred = model.predict_classes(Xarr[img_nr].reshape(1,64,64,3))
plt.title("Prediction:" + str(Ypred) + " Actual Class: " + str(Yarr[3]))
plt.figtext(0.2,0.02,"Daisy = 0, Dandelion = 1, Rose = 2, Sunflower = 3, Tulip = 4")
plt.show()

from utils import load_data

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
#load the data
(feature, labels) = load_data()

#perform a train test split on the data
x_train, x_test,y_train,y_test = train_test_split(feature,labels,test_size=0.5)

#set the categories
categories = ['daisy', 'sunflower','rose', 'tulip','dandelion']

#load the trained model
model = tf.keras.models.load_model('mymodel.h5')

#predict on the trained model
prediction = model.predict(x_test)

plt.figure(figsize=(12,12))

#show 12 random images with preiction and actual data
for i in range(12):
    plt.subplot(4,4,i+1)
    plt.imshow(x_test[i])
    #set the label for prediction and actual data
    plt.xlabel('Actual data:'+categories[y_test[i]]+'\n'+'Predicted data:'+categories[np.argmax(prediction[i])])
    plt.xticks([])
plt.show()

#Test how may are guessed correct and wrong
count_correct = 0
count_false = 0
for i in range(len(x_test)):
    if categories[y_test[i]] == categories[np.argmax(prediction[i])]:
        count_correct = count_correct + 1
    else: count_false = count_false + 1
print("Correct Guesses: " + str(count_correct))
print("Wrong Guesses: " + str(count_false))



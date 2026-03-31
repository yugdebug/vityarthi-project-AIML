import tensorflow as tf
   
import numpy as np
import cv2 as cv  

import matplotlib.pyplot as plt

from tensorflow.python.keras.metrics import accuracy


mnist_train_dataset = tf.keras.datasets.mnist

(x_trainin, y_trainin),(x_testin,y_testin)=mnist_train_dataset.load_data() # split the data in training set as a simple tuple

x_trainin = tf.keras.utils.normalize(x_trainin , axis = 1)
x_testin = tf.keras.utils.normalize(x_testin , axis = 1)

#create the Ml layers 
model= tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))#units are the neurons of the given network

model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))

model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_trainin,y_trainin, epochs=3)#As the number of epochs increases beyond 11,chance of overfitting of the model on training data

loss , accuracy  = model.evaluate(x_testin,y_testin)
print("accuracy of the model is ",accuracy)
print("loss is",loss)

#lets test the given images for digits in them
for x in range(1,4):
    # now we are going to read images it with open cv

    img=cv.imread(f'{x}.png')[:,:,0]#all of it and 1st and last one
    img=np.invert(np.array([img])) #invert black to white in images so that the model will see it as it is
    prediction=model.predict(img)
    
    print("----------------")#some space
    print("The ML predicted value is : ",np.argmax(prediction))
    print("----------------")


    plt.imshow(img[0],cmap=plt.cm.binary) 
    plt.show()



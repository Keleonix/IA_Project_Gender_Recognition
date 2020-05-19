import numpy as np
import keras
import matplotlib.pyplot as plt
import glob
import os

from numpy import array
from keras.layers import Dense, Flatten, Input
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras_vggface.vggface import VGGFace

#==========================================================================================#

hidden_dim = 512
nb_class = 1

#==========================================================================================#

def getGender(filename):
    gender = 1
    #A compléter#
    return gender

#==========================================================================================#

X_train = []
X_test = []
y_train = []
y_test = []
count = 0

# print('Pre-processing pictures :')

for filename in glob.glob('./IA_Project_Gender_Recognition/CelebA/Img/*.jpg'):
    im = load_img(filename, target_size=(224, 224))
    im = img_to_array(im)
    im = im.reshape((1, im.shape[0], im.shape[1], im.shape[2]))
    if(count % 10 == 0 | count % 10 == 4):
        X_test.append(im)
        if(getGender(filename) == 1):
            y_test.append(1)
        else:
            y_test.append(0)
    else:
        X_train.append(im)
        if(getGender(filename) == 1):
            y_train.append(1)
        else:
            y_train.append(0)
    count = count + 1

#==========================================================================================#

X_train = array(X_train)
X_test = array(X_test)
X_train = X_train.reshape(X_train.shape[0], 224, 224, 3).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 224, 224, 3).astype('float32')
X_train = X_train / 255
X_test = X_test / 255

# print(len(X_test), len(y_test), len(X_train), len(y_train))

#==========================================================================================#

def model():
    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
    last_layer = vgg_model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(hidden_dim, activation='relu', name='fc6')(x)
    out = Dense(nb_class, activation='softmax', name='fc8')(x)
    model = Model(inputs = vgg_model.input, outputs = out)
    model.compile(optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
    
    for i in range(19):
        model.layers[i].trainable = False

    # for l in model.layers:
    #     print(l.name, l.trainable)
    
    return model

#==========================================================================================#

model = model()
# print(model.summary())
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=16, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: {} \n Error: {}".format(scores[1], 100-scores[1]*100))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#==========================================================================================#
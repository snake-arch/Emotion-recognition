from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Conv2D,Activation,Dropout,BatchNormalization,Flatten,MaxPooling2D
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import l2
import numpy as np
import cv2
import os
import tensorflow as tf
from keras.optimizers import Adam
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

data = []
labels = []
train_path = 'C:/Users/maind/Desktop/mood/train'
ClassesList = os.listdir(train_path)
print("Classes:"+str(ClassesList))
noOfClasses = len(ClassesList)
print("No of classes:"+str(noOfClasses))
print(len(data),len(labels))
print("Importing images=>")
for x in ClassesList:
    myPiclist = os.listdir(train_path+"/"+str(x))
    for y in myPiclist:
        curImg = cv2.imread(train_path+"/"+str(x)+"/"+y)
        curImg=cv2.cvtColor(curImg, cv2.COLOR_BGR2GRAY)
        curImg = cv2.resize(curImg,(32,32))
        curImg = img_to_array(curImg)
        data.append(curImg)
        labels.append(x)
    print(x)
print(len(data),len(labels))

test_path = 'C:/Users/maind/Desktop/mood/test'
ClassesList1 = os.listdir(test_path)
print("Classes:"+str(ClassesList1))
noOfClasses1 = len(ClassesList1)
print("No of classes:"+str(noOfClasses1))
print(len(data),len(labels))
print("Importing images=>")
for x in ClassesList1:
    myPiclist1 = os.listdir(test_path+"/"+str(x))
    for y in myPiclist1:
        curImg1 = cv2.imread(test_path+"/"+str(x)+"/"+y)
        curImg1=cv2.cvtColor(curImg1, cv2.COLOR_BGR2GRAY)
        curImg1 = cv2.resize(curImg1,(32,32))
        curImg1 = img_to_array(curImg1)
        data.append(curImg1)
        labels.append(x)
    print(x)
print(len(data),len(labels))

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data,dtype="float")/255.0
labels = np.array(labels)
print(labels[0:5])
# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels),7)

#computes the total number of examples per class. In this case, classTotals will be
#an array: [3995, 436] for “angry” and “disgusted” and others
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals
print(classTotals,classWeight)
#1.8148596 16.433273   1.7553213  1.  1.4503065  1.4791838 2.246127
classWeight1={0:1.8148596,1: 16.433273,2: 1.7553213,3:  1.,4:1.4503065,5:1.4791838,6: 2.246127}
num_features = 64
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.20, stratify=labels, random_state=42)
print(trainY.shape,trainX.shape)
opt = SGD(lr=0.001, decay=0.001 / 40, momentum=0.9, nesterov=True)
#opt2 = Adam(lr=0.01,decay=0.01/100, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
def myModel():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

    # model = Sequential()
    #
    # model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))
    # model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D((2,2)))
    # model.add(Dropout(0.5))
    #
    # model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D((2,2)))
    # model.add(Dropout(0.5))
    #
    # model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D((2,2)))
    # model.add(Dropout(0.5))
    #
    # model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D((2,2)))
    # model.add(Dropout(0.5))
    #
    # model.add(Flatten())
    #
    # model.add(Dense(2 * 2 * 2 * num_features, activation='relu'))
    # model.add(Dropout(0.4))
    # model.add(Dense(2 * 2 * num_features, activation='relu'))
    # model.add(Dropout(0.4))
    # model.add(Dense(2 * num_features, activation='relu'))
    # model.add(Dropout(0.5))
    #
    # model.add(Dense(7, activation='softmax'))
    # model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])


    # model=Sequential()
    # model.add(Conv2D(32, (3, 3), padding="same",input_shape = inputShape))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization())
    # model.add(Conv2D(32, (3, 3), padding="same"))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D((2,2)))
    # model.add(Dropout(0.5))
    # model.add(Conv2D(64, (3, 3), padding="same"))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization())
    # model.add(Conv2D(64, (3, 3), padding="same"))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(0.5))
    # model.add(Flatten()) #remove for vgg
    # model.add(Dense(256))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization())
    # model.add(Dense(128))              #this too vgg
    # model.add(Activation("relu"))       #this vgg
    # model.add(BatchNormalization())    #this too vgg
    # model.add(Dropout(0.5))
    # model.add(Dense(7))
    # model.add(Activation("softmax"))
    #model.compile(loss="categorical_crossentropy", optimizer="adam",metrics = ["accuracy"]) # set model.compile callbacks param to callbacks to use step rate scheduler
    return model

model= myModel()
print(model.summary())
########################Model training info##########
#print(model.summary())
labelNames = ["angry", "disgusted", "fearful", "happy", "neutral","sad", "surprised"]
#filepath="weights.best.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor="val_loss", mode="min",save_best_only=True, verbose=1)
#callback=[checkpoint]
history=model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=32,epochs=100,verbose=1,shuffle=True,class_weight=classWeight1)
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=labelNames))
model.save("yoyoVGG2.hdf5")

plt.style.use("ggplot")
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
######################################################













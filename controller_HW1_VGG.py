from PyQt5 import QtWidgets, QtGui, QtCore
from keras import models
from numpy.lib.function_base import select
import cv2
import numpy as np
from math import sqrt, exp, pi
from scipy import signal
import Ui_HW1_VGG
import keras
from keras.datasets import cifar10
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, Flatten, BatchNormalization
from keras import regularizers
from keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import SGD
from keras.models import load_model
from keras.utils import np_utils

import matplotlib.pyplot as plt


class MainWindow_VGG(QtWidgets.QMainWindow):
    def __init__(self):
        # in python3, super(Class, self).xxx = super().xxx
        super(MainWindow_VGG, self).__init__()
        self.ui = Ui_HW1_VGG.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.q5_1.clicked.connect(self.sol5_1)
        self.ui.q5_2.clicked.connect(self.sol5_2)
        self.ui.q5_3.clicked.connect(self.sol5_3)
        self.ui.q5_4.clicked.connect(self.sol5_4)
        self.ui.q5_5.clicked.connect(self.sol5_5)

    def sol5_1(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        self.name = ["airplane", "automobile", "bird", "cat",
                     "deer", "dog", "frog", "horse", "ship", "truck"]
        for i in range(9):
            # define subplot
            plt.subplot(330 + 1 + i)
            # plot raw pixel data
            for j in range(10):
                if y_train[i] == j:
                    name = self.name[j]
            plt.axis('off')
            plt.title(name)
            plt.imshow(x_train[i])
            # show the figure
        plt.show()
        self.train_x = x_train
        self.test_x = x_test
        self.train_y = np_utils.to_categorical(y_train, 10)
        self.test_y = np_utils.to_categorical(y_test, 10)

    def sol5_2(self):
        self.batch_size = 32
        self.learning_rate = 0.001
        print("hyperparameters:")
        print("batch size:", self.batch_size)
        print("learning rate:", self.learning_rate)
        print("optimizer: SGD")

    def sol5_3(self):
        weight_decay = 0.0005
        # layer1 32*32*3
        self.model = Sequential()
        self.model.add(Conv2D(64, (3, 3), padding='same',
                              input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))
        # layer2 32*32*64
        self.model.add(Conv2D(64, (3, 3), padding='same',
                              kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # layer3 16*16*64
        self.model.add(Conv2D(128, (3, 3), padding='same',
                              kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))
        # layer4 16*16*128
        self.model.add(Conv2D(128, (3, 3), padding='same',
                              kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # layer5 8*8*128
        self.model.add(Conv2D(256, (3, 3), padding='same',
                              kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))
        # layer6 8*8*256
        self.model.add(Conv2D(256, (3, 3), padding='same',
                              kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))
        # layer7 8*8*256
        self.model.add(Conv2D(256, (3, 3), padding='same',
                              kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # layer8 4*4*256
        self.model.add(Conv2D(512, (3, 3), padding='same',
                              kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))
        # layer9 4*4*512
        self.model.add(Conv2D(512, (3, 3), padding='same',
                              kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))
        # layer10 4*4*512
        self.model.add(Conv2D(512, (3, 3), padding='same',
                              kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # layer11 2*2*512
        self.model.add(Conv2D(512, (3, 3), padding='same',
                              kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))
        # layer12 2*2*512
        self.model.add(Conv2D(512, (3, 3), padding='same',
                              kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))
        # layer13 2*2*512
        self.model.add(Conv2D(512, (3, 3), padding='same',
                              kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))
        # layer14 1*1*512
        self.model.add(Flatten())
        self.model.add(
            Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        # layer15 512
        self.model.add(
            Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        # layer16 512
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))
        self.model.summary()

    def sol5_4(self):
        # sgd = SGD(lr=self.learning_rate, decay=1e-6,
        #           momentum=0.9, nesterov=True)
        # self.model.compile(loss='categorical_crossentropy',
        #                    optimizer=sgd, metrics=['accuracy'])
        # history = self.model.fit(self.train_x, self.train_y, epochs=50, batch_size=self.batch_size,
        #                          validation_split=0.1, verbose=1)
        # plt.plot(history.history['accuracy'])
        # plt.plot(history.history['val_accuracy'])
        # plt.title('Model accuracy')
        # plt.ylabel('Accuracy')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()

        # # 绘制训练 & 验证的损失值
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('Model loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()
        # self.model.save('model.h5')

        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        acc = cv2.imread("Model_accuracy.png")
        plt.imshow(acc)
        plt.subplot(122)
        loss = cv2.imread("Model_loss.png")
        plt.imshow(loss)
        plt.show()

    def sol5_5(self):
        try:
            msg = int(self.ui.lineEdit.text())
            print(msg)
            print(self.test_x[msg])
            plt.figure(1)
            plt.imshow(self.test_x[msg])
            img = self.test_x[msg]
            img = img[None]
            model = load_model('model.h5')
            print("load")
            result = model.predict(img)[0]
            print(result)
            x = range(10)
            plt.figure(2)
            plt.bar(x, result)
            plt.xticks(x, self.name)
            plt.xlabel('x-axis')
            plt.ylabel('y-axis')
            plt.show()

        except:
            print(" ")


if __name__ == '__main__':
    import sys
    print(keras.__version__)
    app = QtWidgets.QApplication(sys.argv)
    win_VGG = MainWindow_VGG()
    win_VGG.show()
    sys.exit(app.exec_())

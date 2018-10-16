import keras
import math
from keras.models import Sequential
from keras.optimizers import SGD,Adam
from keras.layers import GlobalAveragePooling2D, Add, concatenate, Lambda, Input, Layer, Dense, Conv2D 
from keras.layers import MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.regularizers import l2
from keras.utils import np_utils, Sequence
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras.preprocessing.image import apply_transform, transform_matrix_offset_center, flip_axis
# from keras.applications.imagenet_utils import preprocess_input, _obtain_input_shape
from keras.initializers import he_normal
#import matplotlib.pyplot as plt
#from sklearn.metrics import log_loss

# from load_cifar10 import load_cifar10_data
import csv
import cv2
import os
import numpy as np
from scipy.io import loadmat, savemat
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation

def densenet(num_classes, input_shape):
    
    def NiNBlock(kernel, mlps, strides):
        weight_decay=1e-4
        def inner(x):
            l = BatchNormalization()(x)
            l = Conv2D(mlps[0], kernel, strides=strides, padding='same',kernel_initializer=he_normal(),kernel_regularizer=l2(weight_decay),use_bias=False)(l)
            l = Dropout(0.2)(l)
            l = Activation('relu')(l)
            for size in mlps[1:]:
                l = BatchNormalization()(l)
                l = Conv2D(size, 1, strides=[1,1],kernel_initializer=he_normal(),kernel_regularizer=l2(weight_decay),use_bias=False)(l)
                l = Activation('relu')(l)
            return l
        return inner

    def parallelBlock(inputNIN):
        l = Dense(1024)(inputNIN)
        l = Dropout(0.2)(l)
        l = Activation('relu')(l)
        l = Dense(1024)(l)
        l = Dropout(0.2)(l)
        l = Activation('relu')(l) # or sigmoid
        return l

    def get_model(img_rows, img_cols,rgb,noOfParallelBlocks):
        img = Input(shape=(img_rows, img_cols, rgb))
        l1 = NiNBlock(7, [96, 96, 96], [2,2])(img)
        l1 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(l1)
    #     l1 = Dropout(0.7)(l1)

        l2 = NiNBlock(5, [256, 256, 256], [2,2])(l1)
        l2 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(l2)
    #     l2 = Dropout(0.7)(l2)

        l3 = NiNBlock(3, [512, 512, 512], [1,1])(l2)

        l4 = NiNBlock(3, [1024, 1024, 512,384], [1,1])(l3)
        l5 = NiNBlock(3, [512, 512, 512], [2,2])(l4)
        l5 = GlobalAveragePooling2D()(l5)
        out = []
        for i in noOfParallelBlocks:
            out.append(parallelBlock(l5))
        

        model = Model(inputs=img, outputs=out)
        return model
    img_rows = input_shape[0];
    img_cols = input_shape[1];
    rgb = input_shape[2];
    model = get_model(img_rows,img_cols,rgb)
    return model


class TopSequence(Sequence):
# class TopSequence():
    def __init__(self, x, y, batch_size, img_size, test_mode=False):
        print("init")
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.img_size = img_size
        self.test_mode = test_mode
        self.data_size = x.shape[0]
        print(self.data_size)

    def __len__(self):
        print("len")
        return self.data_size // self.batch_size

    def __getitem__(self,batch_idx):
        print("called")
        if (batch_idx + 1) * self.batch_size - 1 >= self.data_size:
            batch_idx = np.random.randint(self.data_size - 1)

        # Create empty arrays to contain batch of images and labels
        batch_images = np.zeros((self.batch_size, self.img_size[0], self.img_size[1], self.img_size[2]))
        batch_top_outer_category = np.zeros((self.batch_size,5), dtype=np.uint32)

        for b in range(self.batch_size):
            img_path = self.x[batch_idx * self.batch_size + b]
            im = cv2.imread(img_path)
            im = cv2.resize(im,(self.img_size[0], self.img_size[1]))
            im = im / 255.0
#             im = augment_image(im)
#             im = im * 255.0

            batch_images[b, :, :, :] = im                                                    #storing image
            batch_top_outer_category[b, :] = np_utils.to_categorical(int(self.y[batch_idx * self.batch_size + b]), 5)
        
        batch_images = preprocess_input(batch_images)
        print(batch_images.shape)
        return batch_images, batch_top_outer_category


img_rows, img_cols = 112, 112
channel = 3
num_classes = 18
batch_size = 256
nb_epoch = 10

X_train = []
X_valid = []
Y_train = []
Y_valid = []

with open("/data/ILRW/top_outer_category_balanced_train.csv") as file_obj:
    reader = csv.DictReader(file_obj, delimiter=',')
    for line in reader:
        X_train.append(line['x'])
        Y_train.append(line['y'])

with open("/data/ILRW/top_outer_category_balanced_test.csv") as file_obj:
    reader = csv.DictReader(file_obj, delimiter=',')
    for line in reader:
        X_valid.append(line['x'])
        Y_valid.append(line['y'])

X_train = np.array(X_train)
X_valid = np.array(X_valid)
Y_train = np.array(Y_train)
Y_valid = np.array(Y_valid)

model = densenet(num_classes, input_shape=(img_rows,img_cols,3))

top_outter_category_layer = Dense(5, activation='softmax', name='top_outer_category_pred')(model.output)

model = Model(model.input,top_outter_category_layer,name="final")
print(model.summary())

ada = Adam(lr = 0.0001,decay=0.0005)
model.compile(loss=['categorical_crossentropy'],optimizer=ada,metrics=['accuracy'])

#model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))

train_seq = TopSequence(X_train, [Y_train,Y_train], batch_size, img_size=[img_rows,img_cols,3])
val_seq = TopSequence(X_valid, [Y_valid,Y_valid], batch_size, img_size=[img_rows,img_cols,3])

model.fit_generator(
    train_seq,
    steps_per_epoch=len(train_seq),
    epochs=nb_epoch,
    callbacks=[checkpoint],
    validation_data=val_seq,
    validation_steps=len(val_seq),
    max_queue_size= 10)



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
from keras.preprocessing.image import apply_transform, transform_matrix_offset_center, flip_axis
from keras.applications.imagenet_utils import preprocess_input, _obtain_input_shape
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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def densenet(num_classes, input_shape):
    

    def block(kernel,channels,strides,x,funcType):
        weight_decay=1e-4
        l = BatchNormalization()(x)
        l = Conv2D(channels, kernel, strides=strides, padding='same',kernel_initializer=he_normal(),kernel_regularizer=l2(weight_decay),use_bias=False)(l)
        l = Activation(funcType)(l)
        return l

    def get_model(img_rows, img_cols,rgb):
        img = Input(shape=(img_rows, img_cols, rgb))
        A1 = block(7,96,[2,2],img,'relu')
        A2 = AveragePooling2D(pool_size=(3,3),strides=(2,2),padding='same')(A1)
        A3 = block(5,256,[2,2],A2,'relu')
        A4 = AveragePooling2D(pool_size=(3,3),strides=(2,2),padding='same')(A3)

        A5 = block(3,512,[1,1],A4,'relu')

        A6 = block(3,1024,[1,1],A5,'relu')

        A7 = block(3,512,[2,2],A6,'relu')
        A8 = GlobalAveragePooling2D()(A7)

        # A8 = Flatten()(A8)
        A9 = Dense(1024)(A8)
        # A10 = BatchNormalization(axis = 1)(A9)
        A11 = Activation('relu')(A9)
        A12 = Dense(1024)(A11)
        # A13 = BatchNormalization(axis = 1)(A12)
        A14 = Activation('relu')(A12)
        model = Model(inputs=img, outputs=A14)

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

    def on_epoch_end(self):
        pass
 
    def __getitem__(self,batch_idx):
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
batch_size = 100
nb_epoch = 150

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

model = densenet(num_classes, input_shape=(112,112,3))

top_outter_category_layer = Dense(5, activation='softmax', name='top_outer_category_pred')(model.output)

model = Model(model.input,top_outter_category_layer,name="final")
print(model.summary())

#opt = SGD(lr=0.001, momentum=0.9, decay=0.0005)
opt = Adam(lr = 0.001,decay=0.0005)
model.compile(loss=['categorical_crossentropy'],optimizer=opt,metrics=['accuracy'])
path = 'model_top_category_p1_p25.h5'
checkpoint = ModelCheckpoint(filepath=path,verbose=1,save_best_only=True)
#model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))

train_seq = TopSequence(X_train, Y_train, batch_size, img_size=[112,112,3])
val_seq = TopSequence(X_valid, Y_valid, batch_size, img_size=[112,112,3])

model.fit_generator(
        train_seq,
        steps_per_epoch=len(train_seq),
        epochs=nb_epoch,
        callbacks=[checkpoint],
        validation_data=val_seq,
        validation_steps=len(val_seq),
        max_queue_size= 10)


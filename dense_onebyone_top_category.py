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
	
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

class Threshold(Layer):
    """Thresholded Rectified Linear Unit.
    It follows:
    `f(x) = x for x > theta`,
    `f(x) = 0 otherwise`.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        theta: float >= 0. Threshold location of activation.
    # References
        - [Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](http://arxiv.org/abs/1402.3337)
    """

    def __init__(self, theta=0.5, **kwargs):
        super(Threshold, self).__init__(**kwargs)
        self.supports_masking = True
        self.theta = K.cast_to_floatx(theta)

    def call(self, inputs, mask=None):
        return K.cast(K.greater(inputs, self.theta), K.floatx())

    def get_config(self):
        config = {'theta': float(self.theta)}
        base_config = super(Threshold, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
		return input_shape


class TopSequence(Sequence):
# class TopSequence():
	def __init__(self, x, y, batch_size, img_size = [32,32,3], test_mode=False):
		self.x = x
		self.y = y
		self.batch_size = batch_size
		self.img_size = img_size
		self.test_mode = test_mode
		self.data_size = x.shape[0]

	def __len__(self):
		return self.data_size // self.batch_size

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
			im = augment_image(im)
			im = im * 255.0

			batch_images[b, :, :, :] = im                                                    #storing image
			batch_top_outer_category[b, :] = np_utils.to_categorical(int(self.y[batch_idx * self.batch_size + b]), 5)

		batch_images = preprocess_input(batch_images)
		
		return batch_images, batch_top_outer_category#[batch_person_direction, batch_top_outer_category, batch_top_outer_neck_style, batch_top_outer_fit, batch_top_outer_formality, batch_top_outer_design, batch_top_outer_length, batch_top_outer_sleeve_length, batch_top_outer_reflection, batch_presence_of_top_inner, batch_top_inner_category, batch_top_inner_neck_style, batch_top_inner_fit, batch_top_inner_formality, batch_top_inner_design, batch_top_inner_length, batch_top_inner_sleeve_length, batch_top_inner_reflection, batch_top_outer_color, batch_top_inner_color]# + batch_top_outer_color + batch_top_inner_color

def augment_image(img, 
                  contrast_range = 0.4, 
                  brightness_range = 0.3,
                  rotation_range = 5, 
                  shift_range = 0.03, 
                  zoom_range = 0.03, 
                  hor_flip = True, 
                  ver_flip = False):
    contrast = ((np.random.rand() * contrast_range) - (contrast_range / 2)) + 1   # To get in the range of 0.75 - 1.25
    brightness = (np.random.rand() * brightness_range) - (brightness_range / 2)   # To get in the range of -0.3 to +0.3

    x = img * contrast + brightness

    # Choose a random set of transformation parameters
    theta = np.pi / 180 * np.random.uniform(-rotation_range, rotation_range)
    tx, ty = np.random.uniform(-shift_range, shift_range, 2)
    zx, zy = np.random.uniform(1 - zoom_range, 1 + zoom_range, 2)
    hor_flip = np.random.random() if hor_flip else 1
    ver_flip = np.random.random() if ver_flip else 1

    # Construct tranformation matrices
    [h, w, c] = x.shape
    img_col_axis = 1
    img_row_axis = 0
    img_channel_axis = 2

    transform_matrix = None
    if theta != 0:
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        tx *= w
        ty *= h
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

    # Apply tranformation matrix
    if transform_matrix is not None:
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = apply_transform(x, transform_matrix, img_channel_axis,
                            fill_mode='nearest', cval=0)

    # Apply mirroring operations
    if hor_flip < 0.5:
        x = flip_axis(x, img_col_axis)

    if ver_flip < 0.5:
        x = flip_axis(x, img_row_axis)

    return x

def vector_binary_crossentropy(y_true, y_pred):
	z = np.zeros((1,len(y_true)), dtype=np.uint32)

	for i in range(len(y_true)):
		z[i] = K.binary_crossentropy(y_true[i], y_pred[i])

	return K.mean(K.sum(z, axis = -1))

def identity_block(input_tensor, kernel_size, filters, stage, block):
	"""
	The identity_block is the block that has no conv layer at shortcut
	Arguments
		input_tensor: input tensor
		kernel_size: defualt 3, the kernel size of middle conv layer at main path
		filters: list of integers, the nb_filters of 3 conv layer at main path
		stage: integer, current stage label, used for generating layer names
		block: 'a','b'..., current block label, used for generating layer names
	"""

	nb_filter1, nb_filter2, nb_filter3 = filters
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(nb_filter2, kernel_size, kernel_size,
					  border_mode='same', name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	x = merge([x, input_tensor], mode='sum')
	x = Activation('relu')(x)
	return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
	"""
	conv_block is the block that has a conv layer at shortcut
	# Arguments
		input_tensor: input tensor
		kernel_size: defualt 3, the kernel size of middle conv layer at main path
		filters: list of integers, the nb_filters of 3 conv layer at main path
		stage: integer, current stage label, used for generating layer names
		block: 'a','b'..., current block label, used for generating layer names
	Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
	And the shortcut should have subsample=(2,2) as well
	"""

	nb_filter1, nb_filter2, nb_filter3 = filters
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2D(nb_filter1, 1, 1, subsample=strides,
					  name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
					  name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	shortcut = Conv2D(nb_filter3, 1, 1, subsample=strides,
							 name=conv_name_base + '1')(input_tensor)
	shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

	x = merge([x, shortcut], mode='sum')
	x = Activation('relu')(x)
	return x

def resnet50_model(img_rows, img_cols, color_type=1, num_classes=None):
	"""
	Resnet 50 Model for Keras

	Model Schema is based on 
	https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

	ImageNet Pretrained Weights 
	https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5

	Parameters:
	  img_rows, img_cols - resolution of inputs
	  channel - 1 for grayscale, 3 for color 
	  num_classes - number of class labels for our classification task
	"""

	# Handle Dimension Ordering for different backends
	global bn_axis
	if K.image_dim_ordering() == 'tf':
	  bn_axis = 3
	  img_input = Input(shape=(img_rows, img_cols, color_type))
	else:
	  bn_axis = 1
	  img_input = Input(shape=(color_type, img_rows, img_cols))

	x = ZeroPadding2D((3, 3))(img_input)
	#x = Conv2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
	x = Conv2D(64, 7, 7, subsample=(1, 1), name='conv1')(x)
	x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
	x = Activation('relu')(x)
	x = MaxPooling2D((3, 3), strides=(2, 2))(x)

	x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

	x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

	x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

	x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
	x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
	x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

	# Fully Connected Softmax Layer
	x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
	x_fc = Flatten()(x_fc)
	x_fc = Dense(1000, activation='softmax', name='fc10')(x_fc)
	# x_fc = Dense(nu, activation='softmax', name='fc1000')(x_fc)

	# Create model
	model = Model(img_input, x_fc)

	return model

def densenet(classes_num, input_shape=None, input_tensor=None, include_top=True, weight_decay=1e-4,
             depth=100, growth_rate=12, compression=0.5):

    def bn_relu(x):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def bottleneck(x):
        channels = growth_rate * 4
        x = bn_relu(x)
        x = Conv2D(channels,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=l2(weight_decay),use_bias=False)(x)
        x = bn_relu(x)
        x = Conv2D(growth_rate,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=l2(weight_decay),use_bias=False)(x)
        return x

    def single(x):
        x = bn_relu(x)
        x = Conv2D(growth_rate,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=l2(weight_decay),use_bias=False)(x)
        return x

    def transition(x, inchannels):
        outchannels = int(inchannels * compression)
        x = bn_relu(x)
        x = Conv2D(outchannels,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=l2(weight_decay),use_bias=False)(x)
        x = AveragePooling2D((2,2), strides=(2, 2))(x)
        return x, outchannels

    def dense_block(x,blocks,nchannels):
        concat = x
        for i in range(blocks):
            x = bottleneck(concat)
            concat = concatenate([x,concat], axis=-1)
            nchannels += growth_rate
        return concat, nchannels

    def dense_layer(x):
        return Dense(classes_num,activation='softmax',kernel_initializer=he_normal(),kernel_regularizer=l2(weight_decay))(x)


    nblocks = (depth - 4) // 6 
    nchannels = growth_rate * 2
    
    # Determine proper input shape
    if int(keras.__version__.split('.')[-1]) >= 8: 
        input_shape = _obtain_input_shape(input_shape,
                                          default_size=32,
                                          min_size=8,
                                          data_format=K.image_data_format(),
                                          require_flatten=include_top)
    else:
        input_shape = _obtain_input_shape(input_shape,
                                          default_size=32,
                                          min_size=8,
                                          data_format=K.image_data_format(),
                                          require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Conv2D(nchannels,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=l2(weight_decay),use_bias=False)(img_input)

    x, nchannels = dense_block(x,nblocks,nchannels)
    x, nchannels = transition(x,nchannels)
    x, nchannels = dense_block(x,nblocks,nchannels)
    x, nchannels = transition(x,nchannels)
    x, nchannels = dense_block(x,nblocks,nchannels)
    x, nchannels = transition(x,nchannels)
    x = bn_relu(x)
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)
    
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    
    model = Model(inputs, x, name='densenet')
    return model


CLASS_DEPTH = 86
NUM_TRAIN_SET = 55876
NUM_TEST_SET = 23947
Y_train_set = np.zeros((NUM_TRAIN_SET, CLASS_DEPTH))
Y_valid_set = np.zeros((NUM_TEST_SET, CLASS_DEPTH))
if __name__ == '__main__':

	img_rows, img_cols = 112, 112
	channel = 3
	num_classes = 18
	batch_size = 32
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

	model = densenet(classes_num=num_classes, input_shape=(32,32,3))
	top_outter_category_layer = Dense(5, activation='softmax', name='top_outer_category_pred')(model.layers[-2].output)
	model = Model(model.input,top_outter_category_layer,name="final")

	def scheduler (epoch):
		if epoch <= 50:
			lr = 0.1
		elif epoch <= 100:
			lr = 0.01
		elif epoch <= 120:
			lr = 0.01
		else:
			lr = 0.001
		print "Learning Rate: ", lr
		return lr

	optimizer = SGD(lr=scheduler(0), momentum=0.9, nesterov=True)	
 
	model.compile(optimizer=optimizer, loss=['categorical_crossentropy'], metrics=['accuracy'])
	path = 'model_top_category_p1_p25.h5'
	checkpoint = ModelCheckpoint(filepath=path,verbose=1,save_best_only=True)
	lr_scheduler = LearningRateScheduler(scheduler)

	train_seq = TopSequence(X_train, Y_train, batch_size, img_size=[32,32,3])
	val_seq = TopSequence(X_valid, Y_valid, batch_size, img_size=[32,32,3])

	history = model.fit_generator(
		train_seq,
		steps_per_epoch=len(train_seq),
		epochs=nb_epoch,
		callbacks=[checkpoint, lr_scheduler],
		validation_data=val_seq,
		validation_steps=len(val_seq),
		max_queue_size= 10)

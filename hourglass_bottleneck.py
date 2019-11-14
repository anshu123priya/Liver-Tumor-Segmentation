# this code is for the data that is generated from data_lits (1)
import random
from keras.layers.core import *
import sys
import cv2
import nibabel as nib
import pickle as cPickle
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
import numpy as np
import scipy.misc
import numpy.random as rng
from PIL import Image, ImageDraw, ImageFont
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Input, concatenate, Add
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D,Convolution2D
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.models import model_from_json,model_from_config,load_model
from keras.optimizers import SGD,RMSprop,adam,Adam
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.preprocessing import image
from keras import backend as K
from keras.initializers import random_uniform, RandomNormal
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from collections import OrderedDict as od
import tensorflow as tf
import os
import math
import copy
#import matplotlib.pyplot as plt
from contextlib2 import redirect_stdout
initilizers = 'glorot_uniform'


############################################## DICE LOSS ###########################################################################################

def dice_coef(y_true, y_pred, smooth=1):
	intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
	return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

	
def dice_coef_loss(y_true, y_pred):
	return 1-dice_coef(y_true, y_pred)


########################################### HOURGLASS #######################################################################################

def hourglass(input_layer):

	print (input_layer.shape) #(112,112,256)
	conv0 = Conv2D(256, (1,1), activation='relu', padding='same', kernel_initializer=initilizers)(input_layer)
	conv0 = BatchNormalization()(conv0)

	conv1 = Conv2D(64, (1, 1), activation='relu', padding='same',kernel_initializer=initilizers)(conv0)
	conv1 = BatchNormalization()(conv1)
	conv1 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer=initilizers)(conv1)
	conv1 = BatchNormalization()(conv1)
	conv1 = Conv2D(256, (1, 1), activation='relu', padding='same',kernel_initializer=initilizers)(conv1)
	conv1 = BatchNormalization()(conv1)
	residual1 = Add()([conv0,conv1])

	pool1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same')(residual1) #56

	branch1 = Conv2D(64, (1, 1), activation='relu', padding='same',kernel_initializer=initilizers)(residual1)
	branch1 = BatchNormalization()(branch1)
	branch1 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer=initilizers)(branch1)
	branch1 = BatchNormalization()(branch1)
	branch1 = Conv2D(256, (1, 1), activation='relu', padding='same',kernel_initializer=initilizers)(branch1)
	branch1 = BatchNormalization()(branch1)
	bresidual1 = Add()([residual1,branch1])

	conv2 = Conv2D(64, (1, 1), activation='relu', padding='same',kernel_initializer=initilizers)(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer=initilizers)(conv2)
	conv2 = BatchNormalization()(conv2)
	conv2 = Conv2D(256, (1, 1), activation='relu', padding='same',kernel_initializer=initilizers)(conv2)
	conv2 = BatchNormalization()(conv2)
	residual2 = Add()([pool1,conv2])

	pool2 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same')(residual2) #28

	branch2 = Conv2D(64, (1, 1), activation='relu', padding='same',kernel_initializer=initilizers)(residual2)
	branch2 = BatchNormalization()(branch2)
	branch2 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer=initilizers)(branch2)
	branch2 = BatchNormalization()(branch2)
	branch2 = Conv2D(256, (1, 1), activation='relu', padding='same',kernel_initializer=initilizers)(branch2)
	branch2 = BatchNormalization()(branch2)
	bresidual2 = Add()([residual2,branch2])


	###########################BOTLLENECK######################################

	conv4 = Conv2D(64, (1, 1), activation='relu', padding='same',kernel_initializer=initilizers)(pool2)
	conv4 = BatchNormalization()(conv4)
	conv4 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer=initilizers)(conv4)
	conv4 = BatchNormalization()(conv4)
	conv4 = Conv2D(256, (1, 1), activation='relu', padding='same',kernel_initializer=initilizers)(conv4)
	conv4 = BatchNormalization()(conv4)
	residual4 = Add()([pool2,conv4])

	##############################################################################

	up2 = Conv2DTranspose(256,(2,2),strides = (2,2), activation = 'relu', padding = 'same',kernel_initializer=initilizers)(residual4)#######(conv5)
	up2 = BatchNormalization()(up2) #56
	add2 = Add()([up2,bresidual2])

	uconv2 = Conv2D(64, (1, 1), activation='relu', padding='same',kernel_initializer=initilizers)(add2)
	uconv2 = BatchNormalization()(uconv2)
	uconv2 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer=initilizers)(uconv2)
	uconv2 = BatchNormalization()(uconv2)
	uconv2 = Conv2D(256, (1, 1), activation='relu', padding='same',kernel_initializer=initilizers)(uconv2)
	uconv2 = BatchNormalization()(uconv2)
	uresidual2 = Add()([add2,uconv2])

	up3 = Conv2DTranspose(256,(2,2),strides = (2,2), activation = 'relu', padding = 'same',kernel_initializer=initilizers)(uresidual2)#######(conv5)
	up3 = BatchNormalization()(up3) #112
	add3 = Add()([up3,bresidual1])

	uconv3 = Conv2D(64, (1, 1), activation='relu', padding='same',kernel_initializer=initilizers)(add3)
	uconv3 = BatchNormalization()(uconv3)
	uconv3 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer=initilizers)(uconv3)
	uconv3 = BatchNormalization()(uconv3)
	uconv3 = Conv2D(256, (1, 1), activation='relu', padding='same',kernel_initializer=initilizers)(uconv3)
	uconv3 = BatchNormalization()(uconv3)
	uresidual3 = Add()([add3,uconv3])

	return uresidual3

################################################## GENERATOR #################################################################################

def gen(input_img):

	################# ENCODER ##################################################
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer=initilizers)(input_img)
	conv1 = BatchNormalization()(conv1)

	conv2 = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer=initilizers)(conv1)
	conv2 = BatchNormalization()(conv2)
	conv3 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer=initilizers)(conv2)#(pool1)
	conv3 = BatchNormalization()(conv3)

	residual1 =Add()([Conv2D(64, (1, 1), activation='relu', padding='same',kernel_initializer=initilizers)(conv1),conv3])
	
	conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer=initilizers)(residual1)
	conv4 = BatchNormalization()(conv4)

	pool2 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same')(conv4)
		
	conv5 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer=initilizers)(pool2)
	conv5 = BatchNormalization()(conv5)

	conv6 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer=initilizers)(conv5)
	conv6 = BatchNormalization()(conv6)
	residual2 =Add()([Conv2D(128, (1, 1), activation='relu', padding='same', name = 'resd2',kernel_initializer=initilizers)(pool2),conv6])

	pool = MaxPooling2D(pool_size=(2, 2),padding='same')(residual2) #112
	
	conv7 = Conv2D(256, (3, 3), activation='relu', padding='same',kernel_initializer=initilizers)(pool)#(pool3)
	conv7 = BatchNormalization()(conv7)

	conv8 = Conv2D(512, (3, 3), activation='relu', padding='same',kernel_initializer=initilizers)(conv7)
	conv8 = BatchNormalization()(conv8)

        ##########################################################################
	hg1 = hourglass(conv8)
	hg2 = hourglass(hg1)
	########################### DECODER ############################################

	up1 = Conv2DTranspose(512,(2,2),strides = (2,2), activation = 'relu', padding = 'same',name='AFTER_HG',kernel_initializer=initilizers)(hg2)
	up1 = BatchNormalization()(up1)

	uconv1 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer=initilizers)(up1)
	uconv1 = BatchNormalization()(uconv1)

	uconv2 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer=initilizers)(uconv1)
	uconv2 = BatchNormalization()(uconv2)

	residual4 =Add()([Conv2D(128, (1, 1), activation='relu', padding='same',kernel_initializer=initilizers)(up1),uconv2])

	up2 = Conv2DTranspose(128,(2,2),strides = (2,2), activation = 'relu', padding = 'same',kernel_initializer=initilizers)(residual4)
	up2 = BatchNormalization()(up2)

	uconv3 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer=initilizers)(up2)
	uconv3 = BatchNormalization()(uconv3)

	uconv4 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer=initilizers)(uconv3)
	uconv4 = BatchNormalization()(uconv4)

	residual5 =Add()([Conv2D(64, (1, 1), activation='relu', padding='same',kernel_initializer=initilizers)(up2),uconv4])

	uconv5 = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer=initilizers)(residual5)
	uconv5 = BatchNormalization()(uconv5)

	uconv6 = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer=initilizers)(uconv5)
	uconv6 = BatchNormalization()(uconv6)

	generator = Conv2D(3, (3, 3), activation='sigmoid',padding='same',kernel_initializer=initilizers)(uconv6)

	return generator

##################################################################################################################################

############I / O
i_s = 256
input_shape = Input(shape = (i_s, i_s, 3))#tensor called input_shape is done

model = Model(input_shape, gen(input_shape))

model.compile(loss='mean_squared_error', optimizer = Adam(lr = 0.01 ))

inChannel = outChannel = 3
resizeTo = 256
train_T1 = np.load("/users/home/dlagroup28/wd/anshu/liver_tumor/train_T1.npy")
train_T2 = np.load ("/users/home/dlagroup28/wd/anshu/liver_tumor/train_T2.npy")
validation_T1 = np.load("/users/home/dlagroup28/wd/anshu/liver_tumor/validation_T1.npy")
validation_T2 = np.load("/users/home/dlagroup28/wd/anshu/liver_tumor/validation_T2.npy")
numEpochs = 200
batch_size = 8
number_of_epoch = []
for jj in range(numEpochs):
	print("Running epoch : %d" % jj)
	
	train_T1, train_T2 = shuffle(train_T1, train_T2)
	print (train_T1.shape , train_T2.shape)	

	num_batches = int(len(train_T1)/batch_size)
	loss_per_epoch = 0

	for batch in range(num_batches):
		batch_train_X = train_T1[batch*batch_size:min((batch+1)*batch_size,len(train_T1)),:]
		batch_train_Y = train_T2[batch*batch_size:min((batch+1)*batch_size,len(train_T2)),:]		
		loss = model.train_on_batch(batch_train_X,batch_train_Y)
		loss_per_epoch = loss_per_epoch + loss
	average_epoch_loss = loss_per_epoch/num_batches
	print ('loss_per_epoch: %f\n' %(average_epoch_loss))	

	validation_T1, validation_T2 = shuffle(validation_T1, validation_T2)
	print (validation_T1.shape , validation_T2.shape)	

	num_batches_test = int(len(validation_T1)/batch_size)
	loss_per_epoch_test = 0

	for batch in range(num_batches_test):
		batch_validation_X = validation_T1[batch*batch_size:min((batch+1)*batch_size,len(validation_T1)),:]
		batch_validation_Y = validation_T2[batch*batch_size:min((batch+1)*batch_size,len(validation_T2)),:]		
		loss = model.test_on_batch(batch_validation_X,batch_validation_Y)
		loss_per_epoch_test = loss_per_epoch_test + loss
	average_epoch_loss_test = loss_per_epoch_test/num_batches
	print ('loss_per_epoch_test: %f\n' %(average_epoch_loss_test))	

	model.save_weights("/users/home/dlagroup28/wd/anshu/liver_tumor/Model_HG/model_epoch_v_5"+ str(jj) +".h5")
	s=rng.randint(len(validation_T1))
	print (s)
	validation_T1=validation_T1[s,:,:,:]
	validation_T2=validation_T2[s,:,:,:]
	validation_T1=validation_T1.reshape(1,256,256,3)
	validation_T2=validation_T2.reshape(1,256,256,3)
	decoded_imgs = model.predict(validation_T1)[0]
	temp = np.zeros([256, 256*3,3])
	temp[:,:256,:1] = validation_T1[0,:,:,:1]
	temp[:,256:512,:1] = validation_T2[0,:,:,:1]
	temp[:,256*2:256*3,:1] = decoded_imgs[:,:,:1]
	temp[:,:,1:2] = temp[:,:,:1]
	temp[:,:,2:3] = temp[:,:,:1]
	temp = temp*255
	scipy.misc.imsave('/users/home/dlagroup28/wd/anshu/liver_tumor/AllResults_HG/recon_epoch_' + str(jj) + 'v5' + '.jpg', temp)

print ("Training Complete")








# This network is used for the data that is generated from data_gen file
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
import json
from contextlib2 import redirect_stdout
initilizers = 'glorot_uniform'



############################################## DICE LOSS ###########################################################################################

def dice_coef(y_true, y_pred, smooth=1):
	intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
	return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_loss(model):	
	def dice_coef_loss(y_true, y_pred):
		return 1-dice_coef(y_true, y_pred)
	return dice_coef_loss


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

	########################### DECODER ############################################

	up1 = Conv2DTranspose(512,(2,2),strides = (2,2), activation = 'relu', padding = 'same',name='AFTER_HG',kernel_initializer=initilizers)(conv8)
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

	generator = Conv2D(1, (3, 3), activation='sigmoid',padding='same',kernel_initializer=initilizers)(uconv6)

	return generator

##################################################################################################################################

############I / O




i_s=256
input_shape = Input(shape = (i_s, i_s, 1))#tensor called input_shape is done

model = Model(input_shape, gen(input_shape))
model.summary()
model.compile(loss = dice_loss(model), optimizer = Adam(0.001), metrics = ['accuracy'] )
model_json = model.to_json()
with open("Results_gen/model.json" , "w") as f:
	json.dump(json.loads(model_json), f)


X=[]
Y=[]

print("Reading Text Files..")
files=['NPY_text_img/volume.txt','NPY_text_img/white_liver.txt']

n=[0,0]
for i in range(2):
	n[i]=[line.rstrip('\n').split(",")[0] for line in open(files[i])]
	print(".")
	
print("loading images..")
for i in range(len(n[0])):
	x = cv2.resize(cv2.imread(n[0][i],0), (256,256))
	y = cv2.resize(cv2.imread(n[1][i],0), (256,256))

	Y.append(y)
	X.append(x)
											
Y=np.asarray(Y,np.float32)/255
Y=Y[:,:,:,np.newaxis]
print Y.shape
X=np.asarray(X,np.float32)/255
X=X[:,:,:,np.newaxis]
print X.shape

print("Data_shuffling..")
X,Y = shuffle(X,Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20)
print("Data_shuffled..")

saveModel2 = "Results_gen/Model_gen.h5"
numEpochs =10
batch_size = 8

loss=[]
val_loss=[]
val_loss_min = 1000

print("starting training")
for epoch in range(numEpochs):

	train_X,train_Y = shuffle(X_train,Y_train)

	print ("Epoch : "+str(epoch+1)+"/"+str(numEpochs))

   	history=model.fit(train_X, [train_Y],  validation_data = (X_test, [Y_test]), epochs = 1 , batch_size = batch_size, verbose = 1)

	loss.append(float(history.history['loss'][0]))	
	val_loss.append(float(history.history['val_loss'][0]))
	
	if val_loss_min >= float(history.history['val_loss'][0]) : 
		val_loss_min = float(history.history['val_loss'][0])		
		model.save_weights(saveModel2)
			
	loss_arr=np.asarray(loss)
	val_loss_arr=np.asarray(val_loss)

	np.savetxt('Results_gen/white_liver.txt',loss_arr)
	np.savetxt('Results_gen/val_white_liver.txt',val_loss_arr)

	l = len(X_test)
	for i in range(l):
		x_test=X_test[i,:,:,:]
		y_test=Y_test[i,:,:,:]
		x_test=x_test.reshape(1,256,256,1)
		y_test=y_test.reshape(1,256,256,1)
		decoded_imgs = model.predict(x_test)

		temp = np.zeros([256, 256*3,3])
		temp[:,:256,:1] = x_test[0,:,:,:1]
		temp[:,256:512,:1] = y_test[0,:,:,:1]
		temp[:,256*2:256*3,:1] = decoded_imgs[0,:,:,:1]
		temp[:,:,1:2] = temp[:,:,:1]
		temp[:,:,2:3] = temp[:,:,:1]
	
		temp = temp*255
		scipy.misc.imsave('results_gen_dice_10ep/results_gen_' + str(epoch+1) + '/' + str(i) + ".jpg", temp)
print("training Done.")



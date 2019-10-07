import random
import sys
import cv2
import nibabel as nib
import numpy as np
import scipy.misc
import numpy.random as rng
from PIL import Image, ImageDraw, ImageFont
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from collections import OrderedDict as od
import os
import math



T1_train_matrix = []
T2_train_matrix = []
T1x_images = []
T2x_images = []
T1y_images = []
T2y_images = []


for j in range(131):
	f = '/users/home/dlagroup28/wd/anshu/liver_tumor/vol_small/volume-'+str(j)+'.nii'
	T1xx_images = []
	T1yx_images = []
	a = nib.load(f)
	a = a.get_data()
	resizeTo, resizeTo, sliced_z = a.shape # 512,512,sliced_z
	print("a", a.shape)
	A = np.resize(a,(259,259,sliced_z))
	z=range(sliced_z)
	for k in range(sliced_z):	
		T1xx_images.append(cv2.Sobel(A[:,:,k],cv2.CV_64F,1,0,ksize=5))
		T1yx_images.append(cv2.Sobel(A[:,:,k],cv2.CV_64F,0,1,ksize=5))
	T1xx_images=np.asarray(T1xx_images)
	T1yx_images=np.asarray(T1yx_images)
	img_vol_min = np.min(A)
	img_vol_max = np.max(A)
	img_vol_x_min = np.min(T1xx_images)
	img_vol_x_max = np.max(T1xx_images)
	img_vol_y_min = np.min(T1yx_images)
	img_vol_y_max = np.max(T1yx_images)

	#Volume normaization of each image.
	A = (A - img_vol_min) / (img_vol_max - img_vol_min) #a.shape == (img_row, slices, img_col : 259, sliced_z, 259)
	T1xx_images = (T1xx_images - img_vol_x_min) / (img_vol_x_max - img_vol_x_min) #T1xx_images.shape = (slices, img_row, img_col : sliced_z, 259, 259)
	T1yx_images = (T1yx_images - img_vol_y_min) / (img_vol_y_max - img_vol_y_min) #T1yx_images.shape = (slices, img_row, img_col : sliced_z, 259, 259)
	
	for some in range(sliced_z):
		T1_train_matrix.append(A[:,:,some]) #T1_train_matrix.shape is made into (sliced_z, 259, 259) from a.shape which was (259, sliced_z, 259)
		T1x_images.append(T1xx_images[some, :, :])
		T1y_images.append(T1yx_images[some, :, :])
		#print ('T1_train_matrix.shape, T1x_images.shape, T1y_images.shape ')
		#print (np.shape(T1_train_matrix), np.shape(T1x_images), np.shape(T1y_images))
		#All there arrays have same shape (sliced_z, 259, 259)

	del(T1xx_images)
	del(T1yx_images)


for j in range(5):
	f = '/users/home/dlagroup28/wd/anshu/liver_tumor/seg_small/segmentation-5'+str(j)+'.nii'
	T2xx_images = []
	T2yx_images=[]
	b = nib.load(f)
	b = b.get_data()
	resizeTo, resizeTo, sliced_z = b.shape
	B = np.resize(b, (259,259,sliced_z))
	for k in range(sliced_z):	
		T2xx_images.append(cv2.Sobel(B[:,:,k],cv2.CV_64F,1,0,ksize=5))
		T2yx_images.append(cv2.Sobel(B[:,:,k],cv2.CV_64F,0,1,ksize=5))
	T2xx_images=np.asarray(T2xx_images)
	T2yx_images=np.asarray(T2yx_images)
	img_vol_min = np.min(B)
	img_vol_max = np.max(B)
	img_vol_x_min = np.min(T2xx_images)
	img_vol_x_max = np.max(T2xx_images)
	img_vol_y_min = np.min(T2yx_images)
	img_vol_y_max = np.max(T2yx_images)

	#Volume normaization of each image.
	B = (B - img_vol_min) / (img_vol_max - img_vol_min)
	T2xx_images = (T2xx_images - img_vol_x_min) / (img_vol_x_max - img_vol_x_min)
	T2yx_images = (T2yx_images - img_vol_y_min) / (img_vol_y_max - img_vol_y_min)
	print ('shape of B, T2xx_images, T2yx_images')
	print (B.shape, T2xx_images.shape, T2yx_images.shape) 
	for some in range(sliced_z):
		T2_train_matrix.append(B[:, :,some ])
		T2x_images.append(T2xx_images[some, :, :])
		T2y_images.append(T2yx_images[some, :, :])
		

print("read both ground and input")


train_T1 = data[:15500,:,:,:]
validation_T1 = data[15500:,:,:,:]
train_T2 = Label[:15500,:,:,:]
validation_T2 = Label[15500:,:,:,:] 

print("Training and Testing data has now following shapes")
print("Length of Training Data = ",train_T1.shape)
print("Length of Training Ground Truth Data = ",train_T2.shape)
print("Length of Testing Data = ",validation_T1.shape)
print("Length of Testing Ground Truth Data = ",validation_T2.shape)

del(data)
del(Label)

validation_T1 = np.array(validation_T1)

validation_T2 = np.array(validation_T2)

train_T1 = np.array(train_T1)

train_T2 = np.array(train_T2)
np.save("/users/home/dlagroup28/wd/anshu/liver_tumor/NPY_256_70imgs/train_T1.npy", train_T1)
np.save("/users/home/dlagroup28/wd/anshu/liver_tumor/NPY_256_70imgs/train_T2.npy", train_T2)
np.save("/users/home/dlagroup28/wd/anshu/liver_tumor/NPY_256_70imgs/validation_T1.npy", validation_T1)
np.save("/users/home/dlagroup28/wd/anshu/liver_tumor/NPY_256_70imgs/validation_T2.npy", validation_T2)

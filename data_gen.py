import numpy as np
import os
import glob
from PIL import Image
import sys
sys.path.append('../')
import nibabel as nib
from PIL import Image
import scipy.misc
import cv2
import math
import copy


outdir_1 ='/users/home/dlagroup28/wd/anshu/liver_tumor/NPY_512_5imgs/volume'
path_images_T1 = '/users/home/dlagroup28/wd/anshu/liver_tumor/volume'
data_list_T1 = os.listdir(path_images_T1)

for name in data_list_T1:
	l = nib.load(path_images_T1 +'/'+name)
	print(l.shape)
	l = l.get_data()  #[:,:,:,k]
	print('  Data shape is ' + str(l.shape) + ' .')
	filename = name.split('.')[0] + '.npy'
	T1 = os.path.join(outdir_1, filename)
	np.save(T1, l)
	print('File ' + name + ' is saved in ' + T1 + ' .')
del(l)


outdir_2 ='/users/home/dlagroup28/wd/anshu/liver_tumor/NPY_512_5imgs/segmentation'
path_images_T2 = '/users/home/dlagroup28/wd/anshu/liver_tumor/segmentation'
data_list_T2 = os.listdir(path_images_T2)

for name in data_list_T2:
	l = nib.load(path_images_T2 +'/'+name)
	print(l.shape)
	l = l.get_data()  #[:,:,:,k]
	print('  Data shape is ' + str(l.shape) + ' .')
	filename = name.split('.')[0] + '.npy'
	T2 = os.path.join(outdir_2, filename)
	np.save(T2, l)
	print('File ' + name + ' is saved in ' + T2 + ' .')
del(l)




distortion_folder = '/users/home/dlagroup28/wd/anshu/liver_tumor/NPY_512_5imgs'
f_text_1 = open(distortion_folder+ "/" + "vol.txt", "w+")
f_text_2 = open(distortion_folder+ "/" + "seg.txt", "w+")

def get_all_images():
	for j in range(131):
		filepath = '/users/home/dlagroup28/wd/anshu/liver_tumor/NPY_512_5imgs/vol_small/volume-'+str(j)+'.npy'
		T1 = []
		t1_npy = np.load(filepath)
		t1 = np.array(t1_npy, dtype = np.float32)
		x,y,z=t1.shape	
		t1=np.moveaxis(t1,2,0)
		T1.append(t1)
		a=np.shape(T1)[0]
		for i in range(z):
			scipy.misc.imsave('/users/home/dlagroup28/wd/anshu/liver_tumor/images/volume/vol' + '_' + str(j+1) + '_' + str(i) + '.jpg', T1[a-1][i])
			f_text_1.write('/users/home/dlagroup28/wd/anshu/liver_tumor/images/volume/vol' + '_' + str(j+1) + '_' + str(i) + '.jpg' +"\n")
	print("All images are saved in images, shape " +str(np.shape(T1)))
	return T1
T1 = get_all_images()



def get_all_images():
	for j in range(131):
		filepaths = '/users/home/dlagroup28/wd/anshu/liver_tumor/NPY_512_5imgs/seg_small/segmentation-' + str(j) + '.npy'
		T2 = []
		t2_npy = np.load(filepaths)
		t2 = np.array(t2_npy, dtype = np.float32)
		x,y,z=t2.shape	
		t2=np.moveaxis(t2,2,0)
		T2.append(t2)
		a=np.shape(T2)[0]
		for i in range(z):
			scipy.misc.imsave('/users/home/dlagroup28/wd/anshu/liver_tumor/images/segmentation/seg' + '_' + str(j+1) + '_' + str(i) + '.jpg', T2[a-1][i])
			f_text_2.write('/users/home/dlagroup28/wd/anshu/liver_tumor/images/segmentation/seg' + '_' + str(j+1) + '_' + str(i) + '.jpg' +"\n")
	print("All images are saved in images, shape " +str(np.shape(T2)))
	return T2
T2 = get_all_images()





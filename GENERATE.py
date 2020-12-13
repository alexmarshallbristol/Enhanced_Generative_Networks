import numpy as np

from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Dropout, Concatenate, Lambda, ReLU, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
import tensorflow as tf
_EPSILON = K.epsilon()

import matplotlib as mpl
# mpl.use('TkAgg') 
# mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import scipy

import math
import glob
import time
import shutil
import os

from pickle import load
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

colours_raw_root = [[250,242,108],
					[249,225,104],
					[247,206,99],
					[239,194,94],
					[222,188,95],
					[206,183,103],
					[181,184,111],
					[157,185,120],
					[131,184,132],
					[108,181,146],
					[105,179,163],
					[97,173,176],
					[90,166,191],
					[81,158,200],
					[69,146,202],
					[56,133,207],
					[40,121,209],
					[27,110,212],
					[25,94,197],
					[34,73,162]]

colours_raw_root = np.flip(np.divide(colours_raw_root,256.),axis=0)
cmp_root = mpl.colors.ListedColormap(colours_raw_root)

print(tf.__version__)

def post_process_scaling(input_array, min_max):
	input_array = (input_array * 2.) - 1.
	input_array[:,0] = (((input_array[:,0]+0.97)/1.94)*(min_max[0][1] - min_max[0][0])+ min_max[0][0])
	input_array[:,1] = (((input_array[:,1]+0.97)/1.94)*(min_max[1][1] - min_max[1][0])+ min_max[1][0])
	input_array[:,2] = (((input_array[:,2]+1.)/1.97)*(min_max[2][1] - min_max[2][0])+ min_max[2][0])
	input_array[:,3] = (((input_array[:,3]+0.97)/1.94)*(min_max[3][1] - min_max[3][0])+ min_max[3][0])
	input_array[:,4] = (((input_array[:,4]+0.97)/1.94)*(min_max[4][1] - min_max[4][0])+ min_max[4][0])
	input_array[:,5] = (((input_array[:,5]+0.97)/1.94)*(min_max[5][1] - min_max[5][0])+ min_max[5][0])
	return input_array

min_max_GAN_paper = np.load('/hdfs/user/am13743/THESIS/MIN_MAXES/min_max_GAN_paper.npy')
min_max_smear = np.load('/hdfs/user/am13743/THESIS/MIN_MAXES/min_max_smear.npy')
min_max_ptparam = np.load('/hdfs/user/am13743/THESIS/MIN_MAXES/min_max_ptparam.npy')

def ptparam_to_pxpy(input_array):
	x = np.expand_dims(input_array[:,0]*np.sin(input_array[:,1]),1)
	y = np.expand_dims(input_array[:,0]*np.cos(input_array[:,1]),1)
	z = np.expand_dims(input_array[:,2],1)
	px = np.expand_dims(input_array[:,3]*np.sin(input_array[:,4]),1)
	py = np.expand_dims(input_array[:,3]*np.cos(input_array[:,4]),1)
	pz = np.expand_dims(input_array[:,5],1)
	input_array = np.concatenate((x,y,z,px,py,pz),axis=1)
	return input_array


generator = load_model('generator.h5')

for i in range(0, 16):
	noise_size = 1000000
	charge_gan = np.random.choice([-1,1],size=(noise_size,1,1),p=[1-0.5,0.5],replace=True)
	aux_gan = np.abs(np.random.normal(0,1,size=(noise_size,1)))
	gen_noise = np.random.normal(0, 1, (int(noise_size), 100))
	theta_gan = np.random.uniform(low=0., high=1., size=(int(noise_size), 1, 1))*1.94-1.+0.03
	theta_pt_gan = np.random.uniform(low=0., high=1., size=(int(noise_size), 1, 1))*1.94-1.+0.03
	images = generator.predict([np.expand_dims(gen_noise,1), np.expand_dims(aux_gan,1), charge_gan, theta_gan, theta_pt_gan])
	images = np.squeeze(images)
	images = images[:,1:]


	images = (images+1.)/2.
	images = post_process_scaling(images,min_max_ptparam)
	images = ptparam_to_pxpy(images)


	np.save('/storage/am13743/gen_raw_%d'%i,images)

for i in range(0, 16):
	noise_size = 1000000
	charge_gan = np.random.choice([-1,1],size=(noise_size,1,1),p=[1-0.5,0.5],replace=True)
	aux_gan = np.abs(np.random.normal(0,1,size=(noise_size,1)))*1.2
	gen_noise = np.random.normal(0, 1, (int(noise_size), 100))
	theta_gan = np.random.uniform(low=0., high=1., size=(int(noise_size), 1, 1))*1.94-1.+0.03
	theta_pt_gan = np.random.uniform(low=0., high=1., size=(int(noise_size), 1, 1))*1.94-1.+0.03
	images = generator.predict([np.expand_dims(gen_noise,1), np.expand_dims(aux_gan,1), charge_gan, theta_gan, theta_pt_gan])
	images = np.squeeze(images)
	images = images[:,1:]


	images = (images+1.)/2.
	images = post_process_scaling(images,min_max_ptparam)
	images = ptparam_to_pxpy(images)


	np.save('/storage/am13743/gen_boost_%d'%i,images)

# plt.figure(figsize=(5*4, 3*4))
# subplot=0
# for i in range(0, 6):
# 	for j in range(i+1, 6):
# 		subplot += 1
# 		plt.subplot(3,5,subplot)
# 		# if subplot == 3: plt.title(iteration)
# 		plt.hist2d(images[:noise_size,i], images[:noise_size,j], bins=50, norm=LogNorm(), cmap=cmp_root)
# 		# plt.xlabel(axis_titles[i])
# 		# plt.ylabel(axis_titles[j])
# plt.subplots_adjust(wspace=0.3, hspace=0.3)
# # plt.savefig('%s%s/CORRELATIONS/Correlations_%d.png'%(working_directory,saving_directory,iteration),bbox_inches='tight')
# plt.savefig('CORRELATIONS_xy.png',bbox_inches='tight')
# plt.close('all')












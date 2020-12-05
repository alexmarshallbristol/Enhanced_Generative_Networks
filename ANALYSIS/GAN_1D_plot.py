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

plt.rc('text', usetex=True)
plt.rcParams['savefig.dpi'] = 100
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rcParams.update({'font.size': 15})

print(tf.__version__)

transformer_directory = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/TRANSFORMERS/'
pre_trained_directory = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/PRE_TRAIN/'

generator = load_model('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/BLUE_CRYSTAL_RESULTS/GAN_1D/Generator_best_ROC_AUC.h5')

trans_1 = load(open('%strans_1.pkl'%transformer_directory, 'rb'))
trans_2 = load(open('%strans_2.pkl'%transformer_directory, 'rb'))
trans_3 = load(open('%strans_3.pkl'%transformer_directory, 'rb'))
trans_4 = load(open('%strans_4.pkl'%transformer_directory, 'rb'))
trans_5 = load(open('%strans_5.pkl'%transformer_directory, 'rb'))
trans_6 = load(open('%strans_6.pkl'%transformer_directory, 'rb'))

def post_process(input_array):
	output_array = np.empty(np.shape(input_array))
	output_array[:,0] = np.squeeze(trans_1.inverse_transform(np.expand_dims(input_array[:,0],1)*7.))
	output_array[:,1] = np.squeeze(trans_2.inverse_transform(np.expand_dims(input_array[:,1],1)*7.))
	output_array[:,2] = np.squeeze(trans_3.inverse_transform(np.expand_dims(input_array[:,2],1)*7.))
	output_array[:,3] = np.squeeze(trans_4.inverse_transform(np.expand_dims(input_array[:,3],1)*7.))
	output_array[:,4] = np.squeeze(trans_5.inverse_transform(np.expand_dims(input_array[:,4],1)*7.))
	output_array[:,5] = np.squeeze(trans_6.inverse_transform(np.expand_dims(input_array[:,5],1)*7.))
	output_array = ((output_array - 0.1) * 2.4) - 1.
	for i in range(0, 6): # Transformers do not work well with extreme values, not an issue once the network is trained a little, but want to get ROC values for young network
		output_array[np.where(np.isnan(output_array[:,i])==True)] = np.sign(input_array[np.where(np.isnan(output_array[:,i])==True)])
	return output_array

noise_size = 50000

noise = np.random.normal(0,1,size=(noise_size, 1, 100))
aux = np.abs(np.random.normal(0,1,size=(noise_size, 1, 1)))
charge_gan = np.random.choice([-1,1],size=(noise_size,1,1),p=[1-0.5,0.5],replace=True)
generated = np.squeeze(generator.predict([noise,aux,charge_gan]))[:,1:]
generated= post_process(generated)

plt.figure(figsize=(5*4,3*4))
subplot = 0
for i in range(0, 6):
	for j in range(i+1, 6):
		subplot += 1
		plt.subplot(3,5,subplot)
		# plt.title('%d'%iteration)
		plt.hist2d(generated[:,i], generated[:,j], bins=75, norm=LogNorm(), range=[[-1,1],[-1,1]], cmap=cmp_root)
# plt.savefig('%s%s/LATENT_DISTRIBUTIONS'%(working_directory,saving_directory),bbox_inches='tight')
# plt.close('all')
plt.show()




# data = np.load('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/DATA/test_data_16.npy')

# X_test, aux_values, aux_values_4D_test = np.split(data, [-5,-1], axis=1)
# # data = np.load('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/DATA/data_0.npy')

# print(np.shape(data))

# aux_values = aux_values[:1000]
# # aux_values = data[:100000,7:-1]

# print(np.shape(aux_values))
	

# plt.figure(figsize=(8,4))
# plt.subplot(1,2,1)
# plt.hist2d(aux_values[:,1], aux_values[:,3], bins=75, norm=LogNorm(), cmap=cmp_root)
# aux_values = AAE_encoder.predict(aux_values)

# print(np.shape(aux_values))

# print(aux_values)

# plt.subplot(1,2,2)
# # plt.hist2d(aux_values[:,1], aux_values[:,3], bins=75, norm=LogNorm(), cmap=cmp_root)
# plt.hist2d(aux_values[:,1], aux_values[:,2], bins=75, norm=LogNorm(), cmap=cmp_root)
# # plt.savefig('%s%s/LATENT_DISTRIBUTIONS'%(working_directory,saving_directory),bbox_inches='tight')
# # plt.close('all')
# plt.show()



# # plt.figure(figsize=(5*4,3*4))
# # subplot = 0
# # for i in range(0, latent_dim):
# # 	for j in range(i+1, latent_dim):
# # 		subplot += 1
# # 		plt.subplot(3,5,subplot)
# # 		# plt.title('%d'%iteration)
# # 		plt.hist2d(aux_values[:,i], aux_values[:,j], bins=75, norm=LogNorm(), range=[[0,5],[0,5]], cmap=cmp_root)
# # 		plt.xlabel(i)
# # 		plt.ylabel(j)
# # # plt.savefig('%s%s/LATENT_DISTRIBUTIONS'%(working_directory,saving_directory),bbox_inches='tight')
# # # plt.close('all')
# # plt.show()















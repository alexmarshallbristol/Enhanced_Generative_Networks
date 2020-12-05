import numpy as np

from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Dropout, Concatenate, Lambda, ReLU, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model, load_model, Sequential
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

# plt.rc('text', usetex=True)
# plt.rcParams['savefig.dpi'] = 100
# plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# plt.rcParams.update({'font.size': 15})

print(tf.__version__)


latent_dim = 4

# AAE_encoder = load_model('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/TRAINING/AAE/ENCODER.h5')
# data = np.load('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/DATA/test_data_16.npy')

AAE_encoder = load_model('/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/TRAINING/AAE/ENCODER.h5')
data = np.load('/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/DATA/test_data_16.npy')



X_test, aux_values, aux_values_4D_test = np.split(data, [-5,-1], axis=1)
# data = np.load('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/DATA/data_0.npy')

print(np.shape(data))
# .astype("float32")
# aux_values = aux_values[:1000]
# aux_values = data[:100000,7:-1]
# 
print(np.shape(aux_values))
	

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.hist2d(aux_values[:,1], aux_values[:,3], bins=75, norm=LogNorm(), cmap=cmp_root)
aux_values_latent = AAE_encoder.predict(aux_values.copy())

print(np.shape(aux_values_latent))

print(aux_values_latent)

plt.subplot(1,2,2)
# plt.hist2d(aux_values[:,1], aux_values[:,3], bins=75, norm=LogNorm(), cmap=cmp_root)
plt.hist2d(aux_values_latent[:,1], aux_values_latent[:,2], bins=75, norm=LogNorm(), cmap=cmp_root)
plt.savefig('test',bbox_inches='tight')
plt.close('all')
# plt.show()


bdt_train_size = 100000
images = aux_values

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)

X_train_sample = np.abs(np.random.normal(0,1,size=(bdt_train_size*2,4)))

real_training_data = np.squeeze(X_train_sample[:bdt_train_size])

real_test_data = np.squeeze(X_train_sample[bdt_train_size:])

fake_training_data = np.squeeze(images[:bdt_train_size])

fake_test_data = np.squeeze(images[bdt_train_size:bdt_train_size*2])

real_training_labels = np.ones(bdt_train_size)

fake_training_labels = np.zeros(bdt_train_size)

total_training_data = np.concatenate((real_training_data, fake_training_data))

total_training_labels = np.concatenate((real_training_labels, fake_training_labels))

clf.fit(total_training_data, total_training_labels)

out_real = clf.predict_proba(real_test_data)

out_fake = clf.predict_proba(fake_test_data)



ROC_AUC_SCORE_curr = roc_auc_score(np.append(np.ones(np.shape(out_real[:,1])),np.zeros(np.shape(out_fake[:,1]))),np.append(out_real[:,1],out_fake[:,1]))


plt.hist([out_real[:,1],out_fake[:,1]], bins = 100,label=['real','gen'], histtype='step')
plt.xlabel('Output of BDT')
plt.legend(loc='upper right')
plt.title(ROC_AUC_SCORE_curr)
# plt.savefig('%s%s/bdt/BDT_P_out_%d.png'%(working_directory,saving_directory,cnt), bbox_inches='tight')
plt.savefig('BDT_out_aux.png', bbox_inches='tight')
plt.close('all')



bdt_train_size = 100000
images = aux_values_latent

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)

X_train_sample = np.abs(np.random.normal(0,1,size=(bdt_train_size*2,4)))

real_training_data = np.squeeze(X_train_sample[:bdt_train_size])

real_test_data = np.squeeze(X_train_sample[bdt_train_size:])

fake_training_data = np.squeeze(images[:bdt_train_size])

fake_test_data = np.squeeze(images[bdt_train_size:bdt_train_size*2])

real_training_labels = np.ones(bdt_train_size)

fake_training_labels = np.zeros(bdt_train_size)

total_training_data = np.concatenate((real_training_data, fake_training_data))

total_training_labels = np.concatenate((real_training_labels, fake_training_labels))

clf.fit(total_training_data, total_training_labels)

out_real = clf.predict_proba(real_test_data)

out_fake = clf.predict_proba(fake_test_data)



ROC_AUC_SCORE_curr = roc_auc_score(np.append(np.ones(np.shape(out_real[:,1])),np.zeros(np.shape(out_fake[:,1]))),np.append(out_real[:,1],out_fake[:,1]))


plt.hist([out_real[:,1],out_fake[:,1]], bins = 100,label=['real','gen'], histtype='step')
plt.xlabel('Output of BDT')
plt.legend(loc='upper right')
plt.title(ROC_AUC_SCORE_curr)
# plt.savefig('%s%s/bdt/BDT_P_out_%d.png'%(working_directory,saving_directory,cnt), bbox_inches='tight')
plt.savefig('BDT_out_latent.png', bbox_inches='tight')
plt.close('all')

# plt.figure(figsize=(5*4,3*4))
# subplot = 0
# for i in range(0, latent_dim):
# 	for j in range(i+1, latent_dim):
# 		subplot += 1
# 		plt.subplot(3,5,subplot)
# 		# plt.title('%d'%iteration)
# 		plt.hist2d(aux_values[:,i], aux_values[:,j], bins=75, norm=LogNorm(), range=[[0,5],[0,5]], cmap=cmp_root)
# 		plt.xlabel(i)
# 		plt.ylabel(j)
# # plt.savefig('%s%s/LATENT_DISTRIBUTIONS'%(working_directory,saving_directory),bbox_inches='tight')
# # plt.close('all')
# plt.show()















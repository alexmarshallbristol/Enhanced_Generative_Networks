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

min_max_GAN_paper = np.load('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/MIN_MAXES/min_max_GAN_paper.npy')
min_max_smear = np.load('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/MIN_MAXES/min_max_smear.npy')
min_max_ptparam = np.load('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/MIN_MAXES/min_max_ptparam.npy')

def post_process_scaling(input_array, min_max):
	input_array = (input_array * 2.) - 1.
	input_array[:,0] = (((input_array[:,0]+0.97)/1.94)*(min_max[0][1] - min_max[0][0])+ min_max[0][0])
	input_array[:,1] = (((input_array[:,1]+0.97)/1.94)*(min_max[1][1] - min_max[1][0])+ min_max[1][0])
	input_array[:,2] = (((input_array[:,2]+1.)/1.97)*(min_max[2][1] - min_max[2][0])+ min_max[2][0])
	input_array[:,3] = (((input_array[:,3]+0.97)/1.94)*(min_max[3][1] - min_max[3][0])+ min_max[3][0])
	input_array[:,4] = (((input_array[:,4]+0.97)/1.94)*(min_max[4][1] - min_max[4][0])+ min_max[4][0])
	input_array[:,5] = (((input_array[:,5]+0.97)/1.94)*(min_max[5][1] - min_max[5][0])+ min_max[5][0])
	return input_array
def ptparam_to_pxpy(input_array):
	x = np.expand_dims(input_array[:,0]*np.sin(input_array[:,1]),1)
	y = np.expand_dims(input_array[:,0]*np.cos(input_array[:,1]),1)
	z = np.expand_dims(input_array[:,2],1)
	px = np.expand_dims(input_array[:,3]*np.sin(input_array[:,4]),1)
	py = np.expand_dims(input_array[:,3]*np.cos(input_array[:,4]),1)
	pz = np.expand_dims(input_array[:,5],1)
	input_array = np.concatenate((x,y,z,px,py,pz),axis=1)
	return input_array




Stored_seed_aux_values = np.load('Seed_enhanced_generation.npy')

# print(np.shape(Stored_seed_aux_values))

# mom = np.sqrt(generated[:,3]**2+generated[:,4]**2+generated[:,5]**2)
# mom_t = np.sqrt(generated[:,3]**2+generated[:,4]**2)

# plt.figure(figsize=(6, 4))
# ax = plt.subplot(1,1,1)
# plt.hist2d(mom, mom_t, bins=100, norm=LogNorm(), cmap=cmp_root, range=[[0,400],[0,7]],vmin=1)
# plt.xlabel('Momentum (GeV)')
# plt.ylabel('Transverse Momentum (GeV)')
# plt.grid(color='k',linestyle='--',alpha=0.4)
# plt.tight_layout()
# plt.text(0.95, 0.95,'Enhanced Training Distribution',
#      horizontalalignment='right',
#      verticalalignment='top',
#      transform = ax.transAxes, fontsize=15)
# plt.savefig('Correlations_GEN_ENHANCED_PPT_2.png')
# plt.close('all')



# quit()


# generator = load_model('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/BLUE_CRYSTAL_RESULTS/GAN_4D/generator.h5')
generator = load_model('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/BLUE_CRYSTAL_RESULTS/GAN_4D/Generator_best_ROC_AUC.h5')
aux_values = Stored_seed_aux_values[:,-5:-1]
noise_size = 100000
training = Stored_seed_aux_values[:noise_size,1:7]
list_for_np_choice = np.arange(np.shape(aux_values)[0])
random_indicies = np.random.choice(list_for_np_choice, size=int(noise_size*1.2), replace=True)
aux = aux_values[random_indicies]
theta_gan = np.random.uniform(low=0., high=1., size=(int(noise_size*1.2), 1, 1))*1.94-1.+0.03
theta_pt_gan = np.random.uniform(low=0., high=1., size=(int(noise_size*1.2), 1, 1))*1.94-1.+0.03
noise = np.random.normal(0,1,size=(int(noise_size*1.2), 1, 100))
aux = np.expand_dims(aux,1)

# frac = 0.02
# aux[:int(frac*np.shape(aux)[0])][:,:,2] = aux[:int(frac*np.shape(aux)[0])][:,:,2]*1.3

charge_gan = np.random.choice([-1,1],size=(int(noise_size*1.2),1,1),p=[1-0.5,0.5],replace=True)


generated = np.squeeze(generator.predict([noise,aux,charge_gan,theta_gan,theta_pt_gan]))[:,1:]
generated = (generated + 1.)/2.
generated = post_process_scaling(generated, min_max_ptparam)
generated = ptparam_to_pxpy(generated)
generated = generated[np.where(np.sqrt(generated[:,3]**2+generated[:,4]**2+generated[:,5]**2)<400)]
generated = generated[:noise_size]


plt.figure(figsize=(10,4))
ax = plt.subplot(1,2,1)
plt.hist2d(np.sqrt(training[:,3]**2+training[:,4]**2+training[:,5]**2),np.sqrt(training[:,3]**2+training[:,4]**2),bins=100,range=[[0,400],[0,6]],norm=LogNorm(),cmap=cmp_root)
plt.ylabel('Transverse Momentum (GeV)', fontsize=15)
plt.xlabel('Momentum (GeV)', fontsize=15)
plt.text(0.95, 0.95,'Full Simulation',
     horizontalalignment='right',
     verticalalignment='top',
     transform = ax.transAxes, fontsize=15)

ax = plt.subplot(1,2,2)
plt.hist2d(np.sqrt(generated[:,3]**2+generated[:,4]**2+generated[:,5]**2),np.sqrt(generated[:,3]**2+generated[:,4]**2),bins=100,range=[[0,400],[0,6]],norm=LogNorm(),cmap=cmp_root)
plt.ylabel('Transverse Momentum (GeV)', fontsize=15)
plt.xlabel('Momentum (GeV)', fontsize=15)
plt.text(0.95, 0.95,'Enhanced GAN Generation',
     horizontalalignment='right',
     verticalalignment='top',
     transform = ax.transAxes, fontsize=15)

plt.subplots_adjust(hspace=0.25, wspace=0.25)
plt.savefig('THESIS_PLOTS/GENERATE_ENHANCED.pdf',bbox_inches='tight')
plt.close('all')







# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)

# bdt_train_size = 50000

# real_training_data = np.squeeze(training[:bdt_train_size])

# real_test_data = np.squeeze(training[bdt_train_size:])

# fake_training_data = np.squeeze(generated[:bdt_train_size])

# fake_test_data = np.squeeze(generated[bdt_train_size:])

# real_training_labels = np.ones(bdt_train_size)

# fake_training_labels = np.zeros(bdt_train_size)

# total_training_data = np.concatenate((real_training_data, fake_training_data))

# total_training_labels = np.concatenate((real_training_labels, fake_training_labels))

# clf.fit(total_training_data, total_training_labels)

# out_real = clf.predict_proba(real_test_data)

# out_fake = clf.predict_proba(fake_test_data)


# print('real',np.shape(real_test_data), np.shape(out_real))
# print('fake',np.shape(fake_test_data), np.shape(out_fake))

# # plot triangle with average label in each bin
# plt.hist([out_real[:,1],out_fake[:,1]], bins = 100,label=['real','gen'], histtype='step')
# plt.xlabel('Output of BDT')
# plt.legend(loc='upper right')
# # plt.savefig('%s%s/bdt/BDT_P_out_%d.png'%(working_directory,saving_directory,cnt), bbox_inches='tight')
# plt.savefig('BDT_out_ENHANCED.png', bbox_inches='tight')
# plt.close('all')

# ROC_AUC_SCORE_curr = roc_auc_score(np.append(np.ones(np.shape(out_real[:,1])),np.zeros(np.shape(out_fake[:,1]))),np.append(out_real[:,1],out_fake[:,1]))

# print(ROC_AUC_SCORE_curr)

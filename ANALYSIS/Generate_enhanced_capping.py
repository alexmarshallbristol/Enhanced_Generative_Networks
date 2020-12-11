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





# generator = load_model('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/BLUE_CRYSTAL_RESULTS/GAN_4D/Generator_best_ROC_AUC.h5')
generator = load_model('/users/am13743/GANs/TRAINING/GAN_4D/Generator_best_ROC_AUC.h5')


noise_size = 1000000

mu_kinematics = np.empty((0,11))

for i in range(0,14):
	noise = np.random.normal(0,1,size=(int(noise_size), 1, 100))
	aux = np.abs(np.random.normal(0,1,size=(int(noise_size), 1, 4)))
	theta_gan = np.random.uniform(low=0., high=1., size=(int(noise_size*1.2), 1, 1))*0.97+0.015
	theta_pt_gan = np.random.uniform(low=0., high=1., size=(int(noise_size*1.2), 1, 1))*0.97+0.015
	charge_gan = np.random.choice([-1,1],size=(int(noise_size),1,1),p=[1-0.5,0.5],replace=True)

	generated = np.squeeze(generator.predict([noise,aux,charge_gan,theta_gan,theta_pt_gan]))[:,1:]
	generated = post_process_scaling(generated, min_max_ptparam)
	generated = ptparam_to_pxpy(generated)

	charge_gan = np.expand_dims(np.squeeze(charge_gan),1)

	generated = np.concatenate((charge_gan, generated, np.squeeze(aux)), axis=1)

	print(i,np.shape(generated))
	# np.save('Generated_data/gen_%d'%i,generated)
	mu_kinematics = np.concatenate((mu_kinematics, generated),axis=0)

# files = glob.glob('Generated_data/gen_*')

# data = np.empty((0,11))

# for file in files:

# 	current = np.load(file)

# 	data = np.append(data, current, axis=0)

# print(np.shape(data))

# np.save('Generated_data/GEN_TOTAL',data)
# quit()

# mu_kinematics = np.load('Generated_data/GEN_TOTAL.npy')

bin_cut_value = 250 # Bear in mind this cuts the of kinematics of every file. 

##############################################################################################################################


indexes = {'pdg':0, 'x':1, 'y':2, 'z':3, 'px':4, 'py':5, 'pz':6, 'xy_aux':-4, 'z_aux':-3, 'pxpy_aux':-2, 'pz_aux':-1 }


Stored_seed_kinematics = np.empty((0,7))
Stored_seed_aux_values = np.empty((0,4))


# mu_kinematics = np.load('Generated_data/GEN_TOTAL.npy')

print('START',np.shape(mu_kinematics))

mom = np.sqrt(mu_kinematics[:,indexes['px']]**2+mu_kinematics[:,indexes['py']]**2+mu_kinematics[:,indexes['pz']]**2)
mom_t = np.sqrt(mu_kinematics[:,indexes['px']]**2+mu_kinematics[:,indexes['py']]**2)

hist = np.histogram2d(mom, mom_t, bins=50, range=[[0,400],[0,6]])

where_over = np.where(hist[0]>bin_cut_value)

digi = [np.digitize(mom, hist[1]), np.digitize(mom_t, hist[2])]

to_delete = np.empty(0)
for bin_i in range(0, np.shape(np.where(hist[0]>bin_cut_value))[1]):

	where_digi = np.where((digi[0]==where_over[0][bin_i]+1)&(digi[1]==where_over[1][bin_i]+1))

	to_delete = np.append(to_delete, where_digi[0][bin_cut_value:])

mu_kinematics = np.delete(mu_kinematics, to_delete, axis=0)

print(np.shape(mu_kinematics))

Stored_seed_kinematics = np.append(Stored_seed_kinematics, mu_kinematics[:,:7], axis=0)
Stored_seed_aux_values = np.append(Stored_seed_aux_values, mu_kinematics[:,-4:], axis=0)


mom = np.sqrt(Stored_seed_kinematics[:,indexes['px']]**2+Stored_seed_kinematics[:,indexes['py']]**2+Stored_seed_kinematics[:,indexes['pz']]**2)
mom_t = np.sqrt(Stored_seed_kinematics[:,indexes['px']]**2+Stored_seed_kinematics[:,indexes['py']]**2)

plt.figure(figsize=(7,4))
plt.title('TESTING SCRIPT IS WORKING')
plt.hist2d(mom, mom_t, bins=100, norm=LogNorm(),range=[[0,400],[0,6]])
plt.colorbar()
plt.savefig('Enhanced_GEN_distribution_CAPPING.png')
plt.close('all')

plt.figure(figsize=(7,4))
plt.title('TESTING SCRIPT IS WORKING')
plt.hist2d(mom, mom_t, bins=100, norm=LogNorm())
plt.colorbar()
plt.savefig('Enhanced_GEN_distribution_CAPPING_nolim.png')
plt.close('all')

np.save('Stored_seed_aux_values_CAPPING',Stored_seed_aux_values)



aux_values = Stored_seed_aux_values

noise_size = 100000

list_for_np_choice = np.arange(np.shape(aux_values)[0])

random_indicies = np.random.choice(list_for_np_choice, size=int(noise_size*1.2), replace=True)
aux = aux_values[random_indicies]
theta_gan = np.random.uniform(low=0., high=1., size=(int(noise_size*1.2), 1, 1))*0.97+0.015
theta_pt_gan = np.random.uniform(low=0., high=1., size=(int(noise_size*1.2), 1, 1))*0.97+0.015

noise = np.random.normal(0,1,size=(int(noise_size*1.2), 1, 100))
# aux = np.abs(np.random.normal(0,1,size=(noise_size, 1, 4)))
aux = np.expand_dims(aux,1)

aux[:,:,2] = np.power((aux[:,:,2] + 1.),1.15) - 1.


charge_gan = np.random.choice([-1,1],size=(int(noise_size*1.2),1,1),p=[1-0.5,0.5],replace=True)
generated = np.squeeze(generator.predict([noise,aux,charge_gan,theta_gan,theta_pt_gan]))[:,1:]
# generated = post_process(generated)
generated = post_process_scaling(generated, min_max_ptparam)
generated = ptparam_to_pxpy(generated)

generated = generated[np.where(np.sqrt(generated[:,3]**2+generated[:,4]**2+generated[:,5]**2)<400)]

generated = generated[:noise_size]

plt.figure(figsize=(5*4,3*4))
subplot = 0
for i in range(0, 6):
	for j in range(i+1, 6):
		subplot += 1
		plt.subplot(3,5,subplot)
		plt.hist2d(generated[:,i], generated[:,j], bins=75, norm=LogNorm(), cmap=cmp_root)
plt.savefig('Correlations_GEN_ENHANCED_2',bbox_inches='tight')
plt.close('all')



mom = np.sqrt(generated[:,3]**2+generated[:,4]**2+generated[:,5]**2)
mom_t = np.sqrt(generated[:,3]**2+generated[:,4]**2)

plt.figure(figsize=(6, 4))
ax = plt.subplot(1,1,1)
plt.hist2d(mom, mom_t, bins=100, norm=LogNorm(), cmap=cmp_root, range=[[0,400],[0,7]],vmin=1)
plt.xlabel('Momentum (GeV)')
plt.ylabel('Transverse Momentum (GeV)')
plt.grid(color='k',linestyle='--',alpha=0.4)
plt.tight_layout()
plt.text(0.95, 0.95,'Enhanced Training Distribution',
     horizontalalignment='right',
     verticalalignment='top',
     transform = ax.transAxes, fontsize=15)
plt.savefig('Correlations_GEN_ENHANCED_PPT_2.png')
plt.close('all')


quit()
# generated = generated[np.where(np.sqrt(generated[:,3]**2+generated[:,4]**2+generated[:,5]**2)<400)] # cut on 400 GeV

# generated = generated[:noise_size]




from sklearn.neighbors import NearestNeighbors
print('neighbors begining fit')
size = 100000
neighbors = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(generated)

print('neighbors begining query')


distances, indices = neighbors.kneighbors(Stored_seed_aux_values[:size,1:7])

print(np.shape(indices))

print(np.shape(aux[indices]))

seed_aux_from_nearestneighbour = np.expand_dims(np.squeeze(aux[indices]),1)

np.save('seed_aux_from_nearestneighbour',seed_aux_from_nearestneighbour)

quit()




plt.figure(figsize=(5*4,3*4))
subplot = 0
for i in range(0, 6):
	for j in range(i+1, 6):
		subplot += 1
		plt.subplot(3,5,subplot)
		plt.hist2d(generated[:,i], generated[:,j], bins=75, norm=LogNorm(), cmap=cmp_root)
plt.savefig('Correlations_GEN_ENHANCED',bbox_inches='tight')
plt.close('all')



mom = np.sqrt(generated[:,3]**2+generated[:,4]**2+generated[:,5]**2)
mom_t = np.sqrt(generated[:,3]**2+generated[:,4]**2)

plt.figure(figsize=(6, 4))
ax = plt.subplot(1,1,1)
plt.hist2d(mom, mom_t, bins=100, norm=LogNorm(), cmap=cmp_root, range=[[0,400],[0,7]],vmin=1)
plt.xlabel('Momentum (GeV)')
plt.ylabel('Transverse Momentum (GeV)')
plt.grid(color='k',linestyle='--',alpha=0.4)
plt.tight_layout()
plt.text(0.95, 0.95,'Enhanced Training Distribution',
     horizontalalignment='right',
     verticalalignment='top',
     transform = ax.transAxes, fontsize=15)
plt.savefig('Correlations_GEN_ENHANCED_PPT.png')
plt.close('all')

plt.figure(figsize=(6, 4))
ax = plt.subplot(1,1,1)
plt.hist2d(mom, mom_t, bins=100, norm=LogNorm(), cmap=cmp_root, range=[[0,1000],[0,7]],vmin=1)
plt.xlabel('Momentum (GeV)')
plt.ylabel('Transverse Momentum (GeV)')
plt.grid(color='k',linestyle='--',alpha=0.4)
plt.tight_layout()
plt.text(0.95, 0.95,'Enhanced Training Distribution',
     horizontalalignment='right',
     verticalalignment='top',
     transform = ax.transAxes, fontsize=15)
plt.savefig('Correlations_GEN_ENHANCED_PPT_nolim.png')
plt.close('all')



# plt.figure(figsize=(5*4,3*4))
# subplot = 0
# for i in range(0, 6):
# 	for j in range(i+1, 6):
# 		subplot += 1
# 		plt.subplot(3,5,subplot)
# 		plt.hist2d(training[:,i], training[:,j], bins=75, norm=LogNorm(), range=[[-1,1],[-1,1]], cmap=cmp_root)
# plt.savefig('Correlations_TRAIN',bbox_inches='tight')
# plt.close('all')



clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)

bdt_train_size = 50000

real_training_data = np.squeeze(training[:bdt_train_size])

real_test_data = np.squeeze(training[bdt_train_size:])

fake_training_data = np.squeeze(generated[:bdt_train_size])

fake_test_data = np.squeeze(generated[bdt_train_size:])

real_training_labels = np.ones(bdt_train_size)

fake_training_labels = np.zeros(bdt_train_size)

total_training_data = np.concatenate((real_training_data, fake_training_data))

total_training_labels = np.concatenate((real_training_labels, fake_training_labels))

clf.fit(total_training_data, total_training_labels)

out_real = clf.predict_proba(real_test_data)

out_fake = clf.predict_proba(fake_test_data)


print('real',np.shape(real_test_data), np.shape(out_real))
print('fake',np.shape(fake_test_data), np.shape(out_fake))

# plot triangle with average label in each bin
plt.hist([out_real[:,1],out_fake[:,1]], bins = 100,label=['real','gen'], histtype='step')
plt.xlabel('Output of BDT')
plt.legend(loc='upper right')
# plt.savefig('%s%s/bdt/BDT_P_out_%d.png'%(working_directory,saving_directory,cnt), bbox_inches='tight')
plt.savefig('BDT_out_ENHANCED.png', bbox_inches='tight')
plt.close('all')

ROC_AUC_SCORE_curr = roc_auc_score(np.append(np.ones(np.shape(out_real[:,1])),np.zeros(np.shape(out_fake[:,1]))),np.append(out_real[:,1],out_fake[:,1]))

print(ROC_AUC_SCORE_curr)

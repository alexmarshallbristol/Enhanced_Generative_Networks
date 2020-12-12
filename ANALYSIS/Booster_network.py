import numpy as np

from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Dropout, Concatenate, Lambda, ReLU, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import activations
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

def _loss_generator(y_true, y_pred):
	y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
	out = -(K.log(y_pred))
	return K.mean(out, axis=-1)

def split_tensor(index, x):
    return Lambda(lambda x : x[:,:,index])(x)

print(tf.__version__)

Seed_enhanced_generation = np.load('Seed_enhanced_generation.npy')

min_max_ptparam = np.load('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/MIN_MAXES/min_max_ptparam.npy')

# generator = load_model('/Users/am13743/generator.h5')
generator = load_model('/Users/am13743/Generator_best_ROC_AUC.h5')


def pre_process_scaling(input_array, min_max):
	for index in [0,1,3,4,5]:
		range_i = min_max[index][1] - min_max[index][0]
		input_array[:,index] = ((input_array[:,index] - min_max[index][0])/range_i) * 1.94 - 0.97
	for index in [2]:
		range_i = min_max[index][1] - min_max[index][0]
		input_array[:,index] = ((input_array[:,index] - min_max[index][0])/range_i) * 1.97 - 1
	input_array = (input_array + 1.)/2.
	return input_array

def pxpy_to_ptparam(input_array):
	r = np.expand_dims(np.sqrt(input_array[:,0]**2+input_array[:,1]**2),1)
	theta = np.expand_dims(np.arctan2(input_array[:,0],input_array[:,1]),1)
	z = np.expand_dims(input_array[:,2],1)
	pt = np.expand_dims(np.sqrt(input_array[:,3]**2+input_array[:,4]**2),1)
	pt_theta = np.expand_dims(np.arctan2(input_array[:,3],input_array[:,4]),1)
	pz = np.expand_dims(input_array[:,5],1)
	input_array = np.concatenate((r,theta,z,pt,pt_theta,pz),axis=1)
	return input_array

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

inital_plots = True

if inital_plots == True:
	plt.figure(figsize=(5*4, 3*4))
	subplot=0
	for i in range(0, 6):
		for j in range(i+1, 6):
			subplot += 1
			plt.subplot(3,5,subplot)
			plt.hist2d(Seed_enhanced_generation[:,i+1], Seed_enhanced_generation[:,j+1], bins=50, norm=LogNorm(), cmap=cmp_root)
	plt.subplots_adjust(wspace=0.3, hspace=0.3)
	plt.savefig('Booster_network/CORRELATIONS_TRAIN.png',bbox_inches='tight')
	plt.close('all')

	mom = np.sqrt(Seed_enhanced_generation[:,4]**2+Seed_enhanced_generation[:,5]**2+Seed_enhanced_generation[:,6]**2)
	mom_t = np.sqrt(Seed_enhanced_generation[:,4]**2+Seed_enhanced_generation[:,5]**2)


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
	plt.savefig('Booster_network/CORRELATIONS_TRAIN_PPT.png')
	plt.close('all')

	aux_values = Seed_enhanced_generation[:,-5:-1]

	noise_size = np.shape(aux_values)[0]

	theta_gan = np.random.uniform(low=0., high=1., size=(int(noise_size), 1, 1))*0.97+0.015
	theta_pt_gan = np.random.uniform(low=0., high=1., size=(int(noise_size), 1, 1))*0.97+0.015

	noise = np.random.normal(0,1,size=(int(noise_size), 1, 100))
	aux_values = np.expand_dims(aux_values,1)

	charge_gan = np.random.choice([-1,1],size=(int(noise_size),1,1),p=[1-0.5,0.5],replace=True)
	generated = np.squeeze(generator.predict([noise,aux_values,charge_gan,theta_gan,theta_pt_gan]))[:,1:]

	generated = post_process_scaling(generated, min_max_ptparam)
	generated = ptparam_to_pxpy(generated)

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
	plt.savefig('Booster_network/ZERO.png')
	plt.close('all')


Seed_enhanced_generation[:,1:7] = pxpy_to_ptparam(Seed_enhanced_generation[:,1:7])

Seed_enhanced_generation[:,1:7] = pre_process_scaling(Seed_enhanced_generation[:,1:7], min_max_ptparam)

batch_size = 50

# B_architecture = [100,200,100]
B_architecture = [28,28]

save_interval = 25000

optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.5, decay=0, amsgrad=True)

##############################################################################################################
# Build booster model ...
input_noise = Input(shape=(4,))

H = Dense(int(B_architecture[0]))(input_noise)
H = LeakyReLU(alpha=0.2)(H)
H = BatchNormalization(momentum=0.8)(H)
H = Dropout(0.2)(H)

for layer in B_architecture[1:]:

	H = Dense(int(layer))(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = BatchNormalization(momentum=0.8)(H)
	H = Dropout(0.2)(H)

g_output = Dense(4,activation='relu')(H)

booster = Model(inputs=[input_noise], outputs=[g_output])
##############################################################################################################
booster.summary()
# quit(0)


@tf.function
def train_step(images):

	images_pt, images_pz, aux_values = images[:,:,4], images[:,:,6], images[:,0,7:-1]

	images_pt_pz = tf.concat((images_pt,images_pz),axis=-1)

	noise = tf.random.normal([batch_size, 1, 100])
	# aux
	charge_gan = tf.math.sign(tf.random.normal([batch_size, 1, 1]))
	theta_gan = tf.random.uniform([batch_size, 1, 1])*0.97+0.015
	theta_pt_gan = tf.random.uniform([batch_size, 1, 1])*0.97+0.015

	with tf.GradientTape(persistent=True) as tape:

		aux = tf.expand_dims(booster(aux_values, training=True),1)

		generated_images = tf.squeeze(generator([noise, aux, charge_gan, theta_gan, theta_pt_gan]))

		gen_pt = tf.expand_dims(generated_images[:,4],1)
		gen_pz = tf.expand_dims(generated_images[:,6],1)

		gen_pt_pz = tf.concat((gen_pt,gen_pz),axis=-1)

		tape_loss = tf.keras.losses.mean_squared_error(tf.squeeze(images_pt_pz),tf.squeeze(gen_pt_pz))
		tape_loss = tf.math.reduce_mean(tape_loss)


	grad = tape.gradient(tape_loss, booster.trainable_variables)
	optimizer.apply_gradients(zip(grad, booster.trainable_variables))

	return tape_loss



iteration = -1

loss_list = np.empty((0,2))

for epoch in range(int(1E30)):

	X_train = Seed_enhanced_generation

	# boost_weights = X_train[:,-1]
	# boost_weights = np.squeeze(boost_weights/np.sum(boost_weights))
	# list_for_np_choice = np.arange(np.shape(boost_weights)[0])
	# random_indicies_order = np.random.choice(list_for_np_choice, size=np.shape(X_train)[0], p=boost_weights, replace=True) 
	# X_train = X_train[random_indicies_order]

	X_train = np.take(X_train,np.random.permutation(X_train.shape[0]),axis=0,out=X_train)

	# print('Train images shape -',np.shape(X_train))

	list_for_np_choice = np.arange(np.shape(X_train)[0])

	X_train = np.expand_dims(X_train,1).astype("float32")

	train_dataset = (
		tf.data.Dataset.from_tensor_slices(X_train).batch(batch_size,drop_remainder=True).repeat(1)
	)

	for images_for_batch in train_dataset:

		if iteration % 1000 == 0: print('Iteration:',iteration)

		iteration += 1

		loss  = train_step(images_for_batch)

		loss_list = np.append(loss_list, [[iteration, loss]], axis=0)

		if iteration % save_interval == 0 and iteration > 0:

			print('Saving at iteration %d...'%iteration)

			plt.figure(figsize=(4, 4))
			plt.plot(loss_list[:,0], loss_list[:,1])
			plt.ylabel('Loss')
			plt.subplots_adjust(wspace=0.3, hspace=0.3)
			plt.savefig('Booster_network/LOSSES.png',bbox_inches='tight')
			plt.close('all')



			aux_values = Seed_enhanced_generation[:,-5:-1]

			aux_values = booster.predict(aux_values)

			noise_size = np.shape(aux_values)[0]

			theta_gan = np.random.uniform(low=0., high=1., size=(int(noise_size), 1, 1))*0.97+0.015
			theta_pt_gan = np.random.uniform(low=0., high=1., size=(int(noise_size), 1, 1))*0.97+0.015

			noise = np.random.normal(0,1,size=(int(noise_size), 1, 100))
			aux_values = np.expand_dims(aux_values,1)

			charge_gan = np.random.choice([-1,1],size=(int(noise_size),1,1),p=[1-0.5,0.5],replace=True)
			generated = np.squeeze(generator.predict([noise,aux_values,charge_gan,theta_gan,theta_pt_gan]))[:,1:]

			generated = post_process_scaling(generated, min_max_ptparam)
			generated = ptparam_to_pxpy(generated)

			# plt.figure(figsize=(5*4, 3*4))
			# subplot=0
			# for i in range(0, 6):
			# 	for j in range(i+1, 6):
			# 		subplot += 1
			# 		plt.subplot(3,5,subplot)
			# 		plt.title(iteration)
			# 		plt.hist2d(generated[:,i], generated[:,j], bins=50, norm=LogNorm(), cmap=cmp_root)
			# plt.subplots_adjust(wspace=0.3, hspace=0.3)
			# plt.savefig('Booster_network/CORRELATIONS.png',bbox_inches='tight')
			# plt.close('all')

			mom = np.sqrt(generated[:,3]**2+generated[:,4]**2+generated[:,5]**2)
			mom_t = np.sqrt(generated[:,3]**2+generated[:,4]**2)


			plt.figure(figsize=(6, 4))
			ax = plt.subplot(1,1,1)
			plt.hist2d(mom, mom_t, bins=100, norm=LogNorm(), cmap=cmp_root, range=[[0,400],[0,7]],vmin=1)
			plt.xlabel('Momentum (GeV)')
			plt.ylabel('Transverse Momentum (GeV)')
			# plt.title(iteration)
			plt.grid(color='k',linestyle='--',alpha=0.4)
			plt.tight_layout()
			plt.text(0.95, 0.95,'Enhanced Training Distribution',
			     horizontalalignment='right',
			     verticalalignment='top',
			     transform = ax.transAxes, fontsize=15)
			plt.savefig('Booster_network/CORRELATIONS_PPT.png')
			plt.close('all')

			print('Saving done.')



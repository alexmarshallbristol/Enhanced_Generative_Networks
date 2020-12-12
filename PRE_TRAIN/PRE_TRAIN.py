import numpy as np

from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Dropout, Concatenate, Lambda
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
_EPSILON = K.epsilon()

import matplotlib as mpl
mpl.use('TkAgg') 
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import math
import glob
import time
import shutil
import os
import argparse
from pickle import load
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

def split_tensor(index, x):
    return Lambda(lambda x : x[:,:,index])(x)

print(tf.__version__)

working_directory = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/PRE_TRAIN/'
training_directory = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/DATA/'
transformer_directory = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/TRANSFORMERS/'
training_name = 'relu*.npy'
saving_directory = ''
# save_interval = 250000
save_interval = 100000

# working_directory = '/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/PRE_TRAIN/'
# training_directory = '/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/DATA/'
# transformer_directory = '/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/TRANSFORMERS/'
# training_name = 'relu*.npy'
# saving_directory = ''
# save_interval = 50000

close_script_at = save_interval + 5

batch_size = 50


list_of_training_files = glob.glob('%s%s'%(training_directory,training_name))


parser = argparse.ArgumentParser()
parser.add_argument('-c', action='store', dest='config', type=int,
help='network configuration', default=0)
results = parser.parse_args()
configuration = results.config

if configuration == 0:
	D_architecture_aux = [32, 64]
	index_in_GAN_output_vector = 1
	index_of_aux_column = 7
	aux_name = 'r'
	plot_index_x = 1
	plot_index_y = 4
elif configuration == 1:
	D_architecture_aux = [32, 64]
	index_in_GAN_output_vector = 3
	index_of_aux_column = 8
	aux_name = 'z'
	plot_index_x = 1
	plot_index_y = 4
elif configuration == 2:
	D_architecture_aux = [32, 64]
	index_in_GAN_output_vector = 4
	index_of_aux_column = 9
	aux_name = 'pt'
	plot_index_x = 3
	plot_index_y = 4
elif configuration == 3:
	D_architecture_aux = [32, 64]
	index_in_GAN_output_vector = 6
	index_of_aux_column = 10
	aux_name = 'pz'
	plot_index_x = 1
	plot_index_y = 4
elif configuration == 4:
	D_architecture_aux = [64, 128]
	index_in_GAN_output_vector = 'na'
	index_of_aux_column = 11
	aux_name = '4D'
	plot_index_x = 4
	plot_index_y = 6
else:
	print('Broken')
	quit()

print(' ')
print('Initializing networks...')
print(' ')

disc_pt_optimizer = tf.keras.optimizers.Adam(lr=0.0001)

##############################################################################################################
# Build Discriminator pt model ...
d_input = Input(shape=(1,7))

if index_in_GAN_output_vector != 'na':
	H = split_tensor(index_in_GAN_output_vector, d_input)
	H = Flatten()(H)
else:
	H_r = split_tensor(1, d_input)
	H_z = split_tensor(3, d_input)
	H_pt = split_tensor(4, d_input)
	H_pz = split_tensor(6, d_input)

	H = Concatenate(axis=-1)([H_r, H_z, H_pt, H_pz])
	H = Reshape((1,4))(H)


for layer in D_architecture_aux:

	H = Dense(int(layer))(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = Dropout(0.2)(H)

d_output_aux = Dense(1, activation='relu')(H)

discriminator_aux_pt = Model(d_input, d_output_aux)
#############################################################################################################

@tf.function
def train_step(images):

	images, aux_values = images[:,:,:7], images[:,:,index_of_aux_column]

	with tf.GradientTape(persistent=True) as disc_pt_tape:
		pt_aux_values_reco = discriminator_aux_pt(images, training=True)
		disc_pt_loss = tf.keras.losses.mean_squared_error(tf.squeeze(aux_values),tf.squeeze(pt_aux_values_reco))

	grad_disc_pt = disc_pt_tape.gradient(disc_pt_loss, discriminator_aux_pt.trainable_variables)
	disc_pt_optimizer.apply_gradients(zip(grad_disc_pt, discriminator_aux_pt.trainable_variables))

	return disc_pt_loss

start = time.time()

iteration = -1

loss_list = np.empty((0,2))

ROC_AUC_SCORE_list = np.empty((0,3))
ROC_AUC_SCORE_list = np.append(ROC_AUC_SCORE_list, [[0, 1, 0]], axis=0)
best_ROC_AUC = 1

training_time = 0

t0 = time.time()

for epoch in range(int(1E30)):

	for file in list_of_training_files:

		print('Loading initial training file:',file,'...')

		X_train = np.load(file)

		X_train = np.take(X_train,np.random.permutation(X_train.shape[0]),axis=0,out=X_train)

		X_train[:,1:7] = (X_train[:,1:7] * 2.) - 1.
		
		print('Train images shape -',np.shape(X_train))

		list_for_np_choice = np.arange(np.shape(X_train)[0])

		X_train = np.expand_dims(X_train,1).astype("float32")

		train_dataset = (
			tf.data.Dataset.from_tensor_slices(X_train).batch(batch_size,drop_remainder=True).repeat(1)
		)

		for images_for_batch in train_dataset:

			if iteration == close_script_at:
				quit()

			if iteration % 1000 == 0: print('Iteration:',iteration)

			iteration += 1
			
			disc_loss_np_pt = train_step(images_for_batch)

			loss_list = np.append(loss_list, [[iteration, disc_loss_np_pt]], axis=0)

			if iteration % save_interval == 0 and iteration > 0:

				t1 = time.time()

				total = t1-t0

				training_time += total

				print('Saving at iteration %d...'%iteration)

				plt.figure(figsize=(16, 8))
				plt.subplot(2,3,1)
				plt.plot(loss_list[:,0], loss_list[:,1])
				plt.ylabel('Disc Loss Pt')
				plt.subplots_adjust(wspace=0.3, hspace=0.3)
				plt.savefig('%s%s/LOSSES_%s.png'%(working_directory,saving_directory,aux_name),bbox_inches='tight')
				plt.close('all')


				noise_size = 10000
				
				if iteration == 0: noise_size = 10

				random_indicies = np.random.choice(list_for_np_choice, size=(1,int(noise_size)), replace=False)

				aux_pt = X_train[random_indicies[0]][:,0,index_of_aux_column]
				images = X_train[random_indicies[0]][:,:,:7]

				aux_guess = np.squeeze(discriminator_aux_pt.predict(images))

				plt.figure(figsize=(8,4))
				plt.subplot(1,2,1)
				plt.hist2d(aux_pt,aux_guess,bins=75,norm=LogNorm(),cmap=cmp_root)
				plt.ylabel('Guess')
				plt.xlabel('True Aux')
				plt.subplot(1,2,2)
				plt.scatter(images[:5000,0,plot_index_x],images[:5000,0,plot_index_y],c=aux_guess[:5000],cmap=cmp_root)
				plt.ylabel('pz')
				plt.xlabel('pt')
				plt.savefig('%s%s/AUX_RECO_INFO_%s.png'%(working_directory,saving_directory,aux_name),bbox_inches='tight')
				plt.close('all')

				discriminator_aux_pt.save_weights('%s%s/D_AUX_%s_WEIGHTS.h5'%(working_directory,saving_directory,aux_name))

				print('Saving complete.')
				t0 = time.time()


				



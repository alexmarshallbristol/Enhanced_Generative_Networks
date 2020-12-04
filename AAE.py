import numpy as np

from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Dropout, Embedding, Multiply, Activation, Conv2D, ZeroPadding2D, LocallyConnected2D, Concatenate, GRU, Lambda
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.compat.v1.keras.layers import UpSampling2D
from tensorflow.keras import backend as K
_EPSILON = K.epsilon()
import tensorflow as tf

import math

import time

import matplotlib as mpl
mpl.use('TkAgg') 
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import truncnorm

import glob

import time

import shutil

import os

import pandas as pd

import scipy.stats as stats

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

print(tf.__version__)


working_directory = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/TRAINING/'
training_directory = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/DATA/'
transformer_directory = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/TRANSFORMERS/'
training_name = 'data*.npy'
testing_name = 'test*.npy'
saving_directory = 'AAE'
save_interval = 500


# working_directory = '/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/TRAINING/'
# training_directory = '/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/DATA/'
# transformer_directory = '/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/TRANSFORMERS/'
# training_name = 'data*.npy'
# testing_name = 'test*.npy'
# saving_directory = 'AAE'
# save_interval = 25000

batch_size = 50
weight_gauss_loss = 3


input_dim = 4
latent_dim = 4


list_of_training_files = glob.glob('%s%s'%(training_directory,training_name))

list_of_testing_files = glob.glob('%s%s'%(training_directory,testing_name))
X_test = np.load(list_of_testing_files[0])
pdg_info_test, X_test, aux_values_test, aux_values_4D_test = np.split(X_test, [1,-5,-1], axis=1)
list_for_np_choice_test = np.arange(np.shape(X_test)[0])

try:
	files_to_remove = glob.glob('%s%s/*.png'%(working_directory,saving_directory))
	for file_i in files_to_remove:
		os.remove(file_i)
	files_to_remove = glob.glob('%s%s/*.h5'%(working_directory,saving_directory))
	for file_i in files_to_remove:
		os.remove(file_i)
	files_to_remove = glob.glob('%s%s/*.npy'%(working_directory,saving_directory))
	for file_i in files_to_remove:
		os.remove(file_i)
except:
	print('Output directory already clean')


print(' ')
print('Initializing networks...')
print(' ')


autoencoder_input = Input(shape=(input_dim,))
generator_input = Input(shape=(input_dim,))

encoder = Sequential()
encoder.add(Dense(1000, input_shape=(input_dim,)))#, activation='relu'
encoder.add(LeakyReLU())
encoder.add(Dropout(0.2))
encoder.add(Dense(1000))
encoder.add(LeakyReLU())
encoder.add(Dropout(0.2))
encoder.add(Dense(latent_dim, activation='relu'))

decoder = Sequential()
decoder.add(Dense(1000, input_shape=(latent_dim,)))
decoder.add(LeakyReLU())
decoder.add(Dropout(0.2))
decoder.add(Dense(1000))
decoder.add(LeakyReLU())
decoder.add(Dropout(0.2))
decoder.add(Dense(input_dim, activation='relu'))

discriminator = Sequential()
discriminator.add(Dense(1000, input_shape=(latent_dim,)))
discriminator.add(LeakyReLU())
discriminator.add(Dropout(0.2))
discriminator.add(Dense(1000))
discriminator.add(LeakyReLU())
discriminator.add(Dropout(0.2))
discriminator.add(Dense(1, activation='sigmoid'))

latent = encoder(generator_input)
reco = decoder(latent)
autoencoder = Model(inputs=[generator_input], outputs=[reco, latent])


AE_optimizer = tf.keras.optimizers.Adam(lr=1e-4)
disc_optimizer = tf.keras.optimizers.Adam(lr=1e-4)


@tf.function
def train_step(images):


	images_D = tf.squeeze(images[:batch_size])
	images_S = tf.squeeze(images[batch_size:])

	fake_latent = encoder(images_D)

	discriminator_input = tf.concat([fake_latent, tf.math.abs(tf.random.normal([batch_size, latent_dim]))],0)
	labels_D_0 = tf.zeros((batch_size, 1)) 
	labels_D_1 = tf.ones((batch_size, 1))
	discriminator_labels = tf.concat([labels_D_0, labels_D_1],0)

	with tf.GradientTape() as disc_tape:

		out_values = discriminator(discriminator_input, training=True)
		disc_loss = tf.keras.losses.binary_crossentropy(discriminator_labels,out_values)
		disc_loss = tf.math.reduce_mean(disc_loss)

	labels_stacked = tf.ones((batch_size, 1))

	with tf.GradientTape() as S_tape:

		S_reco, S_latent = autoencoder(images_S, training=True)

		stacked_output = discriminator(S_latent)

		S_loss_disc = _loss_generator(labels_stacked,stacked_output)
		S_loss_reco = tf.keras.losses.binary_crossentropy(images_S,S_reco)
		S_loss_disc = tf.math.reduce_mean(S_loss_disc)
		S_loss_reco = tf.math.reduce_mean(S_loss_reco)
		S_loss = S_loss_reco + S_loss_disc*weight_gauss_loss

	grad_S = S_tape.gradient(S_loss, autoencoder.trainable_variables)
	grad_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

	AE_optimizer.apply_gradients(zip(grad_S, autoencoder.trainable_variables))
	disc_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))

	return S_loss, disc_loss


start = time.time()

iteration = -1

for epoch in range(int(1E30)):

	for file in list_of_training_files:

		print('Loading initial training file:',file,'...')

		X_train = np.load(file)

		pdg_info, X_train, aux_values, aux_values_4D = np.split(X_train, [1,-5, -1], axis=1)

		print('Train images shape -',np.shape(X_train))

		list_for_np_choice = np.arange(np.shape(X_train)[0])

		X_train = np.expand_dims(aux_values,1).astype("float32")

		train_dataset = (
			tf.data.Dataset.from_tensor_slices(X_train).batch(2*batch_size,drop_remainder=True).repeat(1)
		)

		for images_for_batch in train_dataset:

			if iteration % 250 == 0: print('Iteration:',iteration)

			iteration += 1

			S_loss_np, disc_loss_np = train_step(images_for_batch)

			if iteration % save_interval == 0 and iteration > 0:

				print('Saving at iteration %d...'%iteration)


				encoder.save('%s%s/ENCODER.h5'%(working_directory,saving_directory))
				decoder.save('%s%s/DECODER.h5'%(working_directory,saving_directory))


				batch_test = 50000


				random_indicies = np.random.choice(list_for_np_choice_test, size=(2,int(batch_test)), replace=False)
				testing_sample = aux_values_test[random_indicies[0]]


				latent = np.squeeze(encoder.predict(testing_sample))
				reconstruction = np.squeeze(decoder.predict(latent))
				plt.figure(figsize=(5*4,3*4))
				subplot = 0
				for i in range(0, latent_dim):
					for j in range(i+1, latent_dim):
						subplot += 1
						plt.subplot(3,5,subplot)
						plt.title('%d'%iteration)
						plt.hist2d(latent[:,i], latent[:,j], bins=75, norm=LogNorm(), range=[[0,5],[0,5]], cmap=cmp_root)
				plt.savefig('%s%s/LATENT_DISTRIBUTIONS'%(working_directory,saving_directory),bbox_inches='tight')
				plt.close('all')


				plt.figure(figsize=(4*3,2*4))
				subplot = 0
				for i in range(0, input_dim):
					subplot += 1
					plt.subplot(2,3,subplot)
					plt.title('%d'%iteration)
					plt.hist2d(reconstruction[:,i], testing_sample[:,i], bins=75, norm=LogNorm(), cmap=cmp_root)
				plt.savefig('%s%s/RECONSTRUCTION'%(working_directory,saving_directory),bbox_inches='tight')
				plt.close('all')


				df_latent = pd.DataFrame(latent)
				data_latent = df_latent.corr()
				plt.figure(figsize=(10,4))
				plt.matshow(data_latent, cmap=cmp_root, vmin=-1, vmax=1)
				for (i, j), z in np.ndenumerate(data_latent):
				    plt.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
				plt.colorbar()
				plt.title('%d'%iteration)
				plt.savefig('%s%s/LATENT_COV_MATRIX'%(working_directory,saving_directory),bbox_inches='tight')
				plt.close('all')


				plt.figure(figsize=(4*3,4*2))
				subplot=0
				for i in range(0, latent_dim):
					subplot += 1
					plt.subplot(2,3,subplot)
					rand_norm = np.random.normal(0,1,size=np.shape(latent[:,i]))
					plt.title('l: %.5f, n: %.5f'%(stats.kstest(latent[:,i],'norm')[1],stats.kstest(rand_norm,'norm')[1]))
					plt.hist(rand_norm,range=[-5,5],bins=51,histtype='step',color='k')
					plt.hist(latent[:,i],range=[-5,5],bins=51,histtype='step',color='r')
				plt.savefig('%s%s/NORMALITY_KSTESTING'%(working_directory,saving_directory),bbox_inches='tight')
				plt.close('all')

				print('Saving complete.')




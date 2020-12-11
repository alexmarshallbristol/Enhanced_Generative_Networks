import numpy as np

from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Dropout, Concatenate, Lambda, ReLU, Activation
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

def sampling(args):
	z_mean, z_log_var = args
	epsilon = K.random_normal(shape=K.shape(z_mean), mean=0, stddev=1)
	return z_mean + K.exp(z_log_var / 2) * epsilon
	
def reco_loss(x, x_decoded_mean):
	# print(x)
	# print(x_decoded_mean)
	# xent_loss = tf.keras.losses.mean_absolute_error(x, x_decoded_mean)
	xent_loss = tf.keras.losses.binary_crossentropy(x, x_decoded_mean)
	# print(xent_loss)
	# quit(0)
	return xent_loss

def reco_loss_boost_pz(x, x_decoded_mean):
	xent_loss_A = tf.keras.losses.mean_squared_error(x[:,0], x_decoded_mean[:,0])
	xent_loss_B = tf.keras.losses.mean_squared_error(x[:,1], x_decoded_mean[:,1])
	xent_loss_C = tf.keras.losses.mean_squared_error(x[:,2], x_decoded_mean[:,2])
	xent_loss_D = tf.keras.losses.mean_squared_error(x[:,3], x_decoded_mean[:,3])
	xent_loss_E = tf.keras.losses.mean_squared_error(x[:,4], x_decoded_mean[:,4])
	xent_loss_F = tf.keras.losses.mean_squared_error(x[:,5], x_decoded_mean[:,5])
	xent_loss = xent_loss_A + xent_loss_B + xent_loss_C*z_boost_factor + xent_loss_D*pt_boost_factor + xent_loss_E + xent_loss_F*pz_boost_factor
	return xent_loss

def kl_loss(z_mean, z_log_var):
	kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
	return kl_loss

def split_tensor(index, x):
	return Lambda(lambda x : x[:,index])(x)

print(tf.__version__)

# working_directory = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/TRAINING/'
# training_directory = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/DATA/'
# transformer_directory = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/TRANSFORMERS/'
# training_name = 'relu*.npy'
# testing_name = 'test_relu*.npy'
# saving_directory = 'VAE'
# save_interval = 1000

# working_directory = '/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/TRAINING/'
# training_directory = '/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/DATA/'
# transformer_directory = '/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/TRANSFORMERS/'
# training_name = 'relu*.npy'
# testing_name = 'test_relu*.npy'
# saving_directory = 'VAE'
# save_interval = 25000


working_directory = 'TRAINING/'
training_directory = '/hdfs/user/am13743/THESIS/DATA/'
transformer_directory = '/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/TRANSFORMERS/'
pre_trained_directory = '/hdfs/user/am13743/THESIS/PRE_TRAIN/'
training_name = 'relu*.npy'
testing_name = 'test_relu*.npy'
saving_directory = 'VAE'
save_interval = 25000


calculate_ROC = True

sample_boosting = False
loss_boosting = False
if loss_boosting == True and sample_boosting == True:
	print('You cant have everything...')
	quit()
boost_pz_reco_loss = False

pz_boost_factor = 2.
pt_boost_factor = 2.
z_boost_factor = 2.

latent_dim = 3
original_dim = 4
kl_factor = 1E-4
reco_factor = 1

batch_size = 50
# E_architecture = [128,128,128]
# D_architecture = [128,128,128]

E_architecture = [250,500,250]
D_architecture = [250,500,250]

def split_tensor(index, x):
	return Lambda(lambda x : x[:,index])(x)


list_of_training_files = glob.glob('%s%s'%(training_directory,training_name))
list_of_testing_files = glob.glob('%s%s'%(training_directory,testing_name))
X_test = np.load(list_of_testing_files[0])
pdg_info_test, X_test, aux_values_test, aux_values_4D_test = np.split(X_test, [1,-5,-1], axis=1)
list_for_np_choice_test = np.arange(np.shape(X_test)[0]) 

# try:
# 	files_to_remove = glob.glob('%s%s/CORRELATIONS/Correlations_*.png'%(working_directory,saving_directory))
# 	for file_i in files_to_remove:
# 		os.remove(file_i)
# except:
# 	print('/CORRELATIONS/ already clean')
# try:
# 	files_to_remove = glob.glob('%s%s/*.png'%(working_directory,saving_directory))
# 	for file_i in files_to_remove:
# 		os.remove(file_i)
# 	files_to_remove = glob.glob('%s%s/*.h5'%(working_directory,saving_directory))
# 	for file_i in files_to_remove:
# 		os.remove(file_i)
# 	files_to_remove = glob.glob('%s%s/*.npy'%(working_directory,saving_directory))
# 	for file_i in files_to_remove:
# 		os.remove(file_i)
# except:
# 	print('Output directory already clean')


print(' ')
print('Initializing networks...')
print(' ')

##############################################################################################################
# Build encoder model ...
input_sample = Input(shape=(original_dim,))
H_theta = Input(shape=(1,))
H_pt_theta = Input(shape=(1,))

input_sample_in = Concatenate(axis=-1)([input_sample,H_theta,H_pt_theta])

H = Dense(int(E_architecture[0]))(input_sample_in)
H = BatchNormalization(momentum=0.8)(H)
H = LeakyReLU(alpha=0.2)(H)
H = Dropout(0.2)(H)

for layer in E_architecture[1:]:
	H = Dense(int(layer))(H)
	H = BatchNormalization(momentum=0.8)(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = Dropout(0.2)(H)

z_mean = Dense(latent_dim)(H)
z_log_var = Dense(latent_dim)(H)

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

encoder = Model(inputs=[input_sample, H_theta, H_pt_theta], outputs=[z, z_mean, z_log_var, H_theta, H_pt_theta])
##############################################################################################################

##############################################################################################################
# Build decoder model ...
input_latent = Input(shape=(latent_dim))
H_theta = Input(shape=(1,))
H_pt_theta = Input(shape=(1,))

input_latent_in = Concatenate()([input_latent,H_theta,H_pt_theta])

H = Dense(int(D_architecture[0]))(input_latent_in)
H = BatchNormalization(momentum=0.8)(H)
H = LeakyReLU(alpha=0.2)(H)
H = Dropout(0.2)(H)

for layer in D_architecture[1:]:
	H = Dense(int(layer))(H)
	H = BatchNormalization(momentum=0.8)(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = Dropout(0.2)(H)

decoded_mean = Dense(original_dim)(H)

H_r = split_tensor(0, H)
H_r = Activation('sigmoid')(H_r)
H_r = Reshape((1,))(H_r)

# H_theta = split_tensor(1, H)
# H_theta = ReLU(max_value=1)(H_theta)
# H_theta = Reshape((1,1))(H_theta)

H_z = split_tensor(2, H)
H_z = Activation('tanh')(H_z)
H_z = ReLU()(H_z)
H_z = Reshape((1,))(H_z)

H_pt = split_tensor(3, H)
H_pt = Activation('tanh')(H_pt)
H_pt = ReLU()(H_pt)
H_pt = Reshape((1,))(H_pt)

# H_pt_theta = split_tensor(4, H)
# H_pt_theta = ReLU(max_value=1)(H_pt_theta)
# H_pt_theta = Reshape((1,1))(H_pt_theta)

H_pz = split_tensor(5, H)
H_pz = Activation('tanh')(H_pz)
H_pz = ReLU()(H_pz)
H_pz = Reshape((1,))(H_pz)

decoded_mean = Concatenate(axis=-1)([H_r, H_theta, H_z, H_pt, H_pt_theta, H_pz])

decoder = Model(inputs=[input_latent, H_theta, H_pt_theta], outputs=[decoded_mean])
##############################################################################################################

input_sample = Input(shape=(original_dim,))
H_theta = Input(shape=(1,))
H_pt_theta = Input(shape=(1,))
z, z_mean, z_log_var, H_theta_d, H_pt_theta_d = encoder([input_sample,H_theta,H_pt_theta])
decoded_mean = decoder([z, H_theta_d, H_pt_theta_d])
vae = Model(inputs=[input_sample, H_theta, H_pt_theta], outputs=[decoded_mean, z_mean, z_log_var])

optimizer = tf.keras.optimizers.Adam(lr=0.0004, beta_1=0.5, decay=0, amsgrad=True)

@tf.function
def train_step(images):

	pdg, images, aux_values_BIN = images[:,:,0], images[:,0,1:7], images[:,:,7:]

	images_r = tf.expand_dims(images[:,0],1)
	images_z = tf.expand_dims(images[:,2],1)
	images_pt = tf.expand_dims(images[:,3],1)
	images_pz = tf.expand_dims(images[:,5],1)

	images = tf.concat([images_r,images_z,images_pt,images_pz],-1)

	theta_gan = tf.random.uniform([batch_size, 1])*0.97+0.015
	theta_pt_gan = tf.random.uniform([batch_size, 1])*0.97+0.015

	images_full = tf.concat([images_r,theta_gan,images_z,images_pt,theta_pt_gan,images_pz],-1)

	with tf.GradientTape() as tape:

		vae_out, vae_z_mean, vae_z_log_var = vae([images,theta_gan,theta_pt_gan])

		vae_reco_loss = reco_loss(images_full, vae_out)
		vae_reco_loss = tf.math.reduce_mean(vae_reco_loss)
		vae_kl_loss = kl_loss(vae_z_mean, vae_z_log_var)
		vae_kl_loss = tf.math.reduce_mean(vae_kl_loss)

		vae_loss = vae_kl_loss*kl_factor + vae_reco_loss*reco_factor

	grad_vae = tape.gradient(vae_loss, vae.trainable_variables)

	optimizer.apply_gradients(zip(grad_vae, vae.trainable_variables))

	return vae_kl_loss, vae_reco_loss


start = time.time()

iteration = -1

loss_list = np.empty((0,3))

axis_titles_post = ['StartX', 'StartY', 'StartZ', 'Px', 'Py', 'Pz']
axis_titles_train = ['r', 'theta', 'z', 'pt', 'pt_theta', 'pz']
axis_titles_boxcox = ['r - BOXCOX', 'theta - QT', 'z - BOXCOX', 'pt - BOXCOX', 'pt_theta - QT', 'pz - BOXCOX']

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

		if iteration == -1:
			plt.figure(figsize=(5*4, 3*4))
			subplot=0
			for i in range(0, 6):
				for j in range(i+1, 6):
					subplot += 1
					plt.subplot(3,5,subplot)
					if subplot == 3: plt.title(iteration)
					plt.hist2d(X_train[:,i+1], X_train[:,j+1], bins=50,range=[[0,1],[0,1]], norm=LogNorm(), cmap=cmp_root)
					plt.xlabel(axis_titles_boxcox[i])
					plt.ylabel(axis_titles_boxcox[j])
			plt.subplots_adjust(wspace=0.3, hspace=0.3)
			plt.savefig('%s%s/TRAIN.png'%(working_directory,saving_directory),bbox_inches='tight')
			plt.close('all')

		print('Train images shape -',np.shape(X_train))

		list_for_np_choice = np.arange(np.shape(X_train)[0])

		X_train = np.expand_dims(X_train,1).astype("float32")

		train_dataset = (
			tf.data.Dataset.from_tensor_slices(X_train).batch(batch_size,drop_remainder=True).repeat(1)
		)

		for images_for_batch in train_dataset:

			if iteration % 250 == 0: print('Iteration:',iteration)

			iteration += 1

			kl_loss_np, reco_loss_np = train_step(images_for_batch)

			loss_list = np.append(loss_list, [[iteration, kl_loss_np, reco_loss_np]], axis=0)

			if iteration % save_interval == 0:

				t1 = time.time()

				total = t1-t0

				training_time += total

				print('Saving at iteration %d...'%iteration)

				decoder.save('%s%s/decoder.h5'%(working_directory,saving_directory))
				encoder.save('%s%s/encoder.h5'%(working_directory,saving_directory))

				# if iteration < 25000:
				plt.figure(figsize=(12, 8))
				plt.subplot(2,3,1)
				plt.plot(loss_list[:,0], loss_list[:,1])
				plt.ylabel('kl Loss')
				plt.subplot(2,3,2)
				plt.plot(loss_list[:,0], loss_list[:,2])
				plt.ylabel('reco Loss')
				plt.subplot(2,3,3)
				plt.plot(loss_list[:,0], loss_list[:,1])
				plt.ylabel('kl Loss')
				plt.yscale('log')
				plt.subplot(2,3,4)
				plt.plot(loss_list[:,0], loss_list[:,2])
				plt.ylabel('reco Loss')
				plt.yscale('log')
				plt.subplot(2,3,5)
				plt.plot(loss_list[:,0], kl_factor*loss_list[:,1]+reco_factor*loss_list[:,2])
				plt.ylabel('TOTAL Loss')
				plt.yscale('log')
				plt.subplots_adjust(wspace=0.3, hspace=0.3)
				plt.savefig('%s%s/LOSSES.png'%(working_directory,saving_directory),bbox_inches='tight')
				plt.close('all')

				# try:
				noise_size = 100000
		
				if iteration == 0: noise_size = 10

				gen_noise = np.random.normal(0, 1, (int(noise_size), latent_dim))
				theta_gan = np.random.uniform(low=0., high=1., size=(int(noise_size), 1))*0.97+0.015
				theta_pt_gan = np.random.uniform(low=0., high=1., size=(int(noise_size), 1))*0.97+0.015
				images = np.squeeze(decoder.predict([gen_noise,theta_gan,theta_pt_gan]))


				random_indicies = np.random.choice(list_for_np_choice_test, size=(1,int(noise_size)), replace=False)
				sample = X_test[random_indicies[0]]


				images_r = np.expand_dims(sample[:,0],1)
				images_theta = np.expand_dims(sample[:,1],1)
				images_z = np.expand_dims(sample[:,2],1)
				images_pt = np.expand_dims(sample[:,3],1)
				images_theta_pt = np.expand_dims(sample[:,4],1)
				images_pz = np.expand_dims(sample[:,5],1)

				sample_in = np.concatenate((images_r,images_z,images_pt,images_pz),axis=-1)

				latent, z_mean_gen, z_log_var_gen, theta_enc, theta_pt_enc = encoder.predict([sample_in,images_theta,images_theta_pt])

				plt.figure(figsize=(2*4, 3*4))
				subplot=0
				for i in range(0, latent_dim):
					for j in range(i+1, latent_dim):
						subplot += 1
						plt.subplot(3,2,subplot)
						plt.title(iteration)
						plt.hist2d(latent[:noise_size,i], latent[:noise_size,j], bins=50,range=[[-5,5],[-5,5]], norm=LogNorm(), cmap=cmp_root)
				plt.subplots_adjust(wspace=0.3, hspace=0.3)
				plt.savefig('%s%s/LATENT_SPACE.png'%(working_directory,saving_directory),bbox_inches='tight')
				plt.close('all')

				reco = decoder.predict([latent, theta_enc, theta_pt_enc])

				plt.figure(figsize=(2*4, 3*4))
				subplot=0
				for i in range(0, 6):
					subplot += 1
					ax = plt.subplot(3,2,subplot)
					plt.title('%.4f'%scipy.stats.pearsonr(sample[:noise_size,i],reco[:noise_size,i])[0])
					plt.hist2d(sample[:noise_size,i], reco[:noise_size,i], bins=50, norm=LogNorm(), cmap=cmp_root)
					lims = [
						np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
						np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
					]
					plt.plot(lims, lims, 'k-')
				plt.subplots_adjust(wspace=0.3, hspace=0.3)
				plt.savefig('%s%s/RECONSTRUCTION.png'%(working_directory,saving_directory),bbox_inches='tight')
				plt.close('all')


				########

				plt.figure(figsize=(3*4, 2*4))
				subplot=0
				for i in range(0, 6):
					subplot += 1
					plt.subplot(2,3,subplot)
					if subplot == 2: plt.title(iteration)
					plt.hist([sample[:noise_size,i], images[:noise_size,i]], bins=50,range=[0,1], label=['Train','GEN'],histtype='step')
					plt.yscale('log')
					plt.title(iteration)
					plt.xlabel(axis_titles_train[i])
					plt.legend()
				plt.subplots_adjust(wspace=0.3, hspace=0.3)
				plt.savefig('%s%s/VALUES.png'%(working_directory,saving_directory),bbox_inches='tight')
				plt.close('all')

				plt.figure(figsize=(5*4, 3*4))
				subplot=0
				for i in range(0, 6):
					for j in range(i+1, 6):
						subplot += 1
						plt.subplot(3,5,subplot)
						if subplot == 3: plt.title(iteration)
						plt.hist2d(images[:noise_size,i], images[:noise_size,j], bins=50,range=[[0,1],[0,1]], norm=LogNorm(), cmap=cmp_root)
						plt.xlabel(axis_titles_train[i])
						plt.ylabel(axis_titles_train[j])
				plt.subplots_adjust(wspace=0.3, hspace=0.3)
				plt.savefig('%s%s/CORRELATIONS.png'%(working_directory,saving_directory),bbox_inches='tight')
				# plt.savefig('%s%s/CORRELATIONS/Correlations_%d.png'%(working_directory,saving_directory,iteration),bbox_inches='tight')
				plt.close('all')


				if iteration > 0 and calculate_ROC == True:
					try:
						clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)

						bdt_train_size = int(np.shape(sample)[0]/2)

						real_training_data = sample[:bdt_train_size]

						real_test_data = sample[bdt_train_size:]

						fake_training_data = np.squeeze(images[:bdt_train_size])

						fake_test_data = np.squeeze(images[bdt_train_size:])

						real_training_labels = np.ones(bdt_train_size)

						fake_training_labels = np.zeros(bdt_train_size)

						total_training_data = np.concatenate((real_training_data, fake_training_data))

						total_training_labels = np.concatenate((real_training_labels, fake_training_labels))

						clf.fit(total_training_data, total_training_labels)

						out_real = clf.predict_proba(real_test_data)

						out_fake = clf.predict_proba(fake_test_data)

						plt.hist([out_real[:,1],out_fake[:,1]], bins = 100,label=['real','gen'], histtype='step')
						plt.xlabel('Output of BDT')
						plt.title(iteration)
						plt.legend(loc='upper right')
						plt.savefig('%s%s/BDT_out.png'%(working_directory,saving_directory), bbox_inches='tight')
						plt.close('all')

						ROC_AUC_SCORE_curr = roc_auc_score(np.append(np.ones(np.shape(out_real[:,1])),np.zeros(np.shape(out_fake[:,1]))),np.append(out_real[:,1],out_fake[:,1]))

						ROC_AUC_SCORE_list = np.append(ROC_AUC_SCORE_list, [[iteration, ROC_AUC_SCORE_curr, training_time]], axis=0)

						if ROC_AUC_SCORE_list[-1][1] < best_ROC_AUC:
							print('Saving best ROC_AUC.')
							decoder.save('%s%s/Decoder_best_ROC_AUC.h5'%(working_directory,saving_directory))
							encoder.save('%s%s/Encoder_best_ROC_AUC.h5'%(working_directory,saving_directory))
							best_ROC_AUC = ROC_AUC_SCORE_list[-1][1]
							shutil.copy('%s%s/CORRELATIONS.png'%(working_directory,saving_directory), '%s%s/BEST_ROC_AUC_Correlations.png'%(working_directory,saving_directory))



						plt.figure(figsize=(8,4))
						plt.title('ROC_AUC_SCORE_list best: %.4f at %d'%(best_ROC_AUC,ROC_AUC_SCORE_list[np.where(ROC_AUC_SCORE_list==best_ROC_AUC)[0][0]][0]))
						plt.plot(ROC_AUC_SCORE_list[:,0],ROC_AUC_SCORE_list[:,1])
						plt.axhline(y=best_ROC_AUC,c='k',linestyle='--')
						plt.axvline(x=ROC_AUC_SCORE_list[np.where(ROC_AUC_SCORE_list==best_ROC_AUC)[0][0]][0],c='k',linestyle='--')
						plt.savefig('%s%s/ROC_progress.png'%(working_directory,saving_directory))
						plt.close('all')
					except:
						print('Roc failed')



				print('Saving complete.')
				# except:
				# 	print('Saving failed at some point and for some unknown reason.')


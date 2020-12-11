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

def _loss_generator(y_true, y_pred):
	y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
	out = -(K.log(y_pred))
	return K.mean(out, axis=-1)

def split_tensor(index, x):
    return Lambda(lambda x : x[:,:,index])(x)

print(tf.__version__)

# working_directory = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/TRAINING/'
# training_directory = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/DATA/'
# transformer_directory = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/TRANSFORMERS/'
# pre_trained_directory = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/PRE_TRAIN/'
# training_name = 'relu*.npy'
# testing_name = 'test_relu*.npy'
# saving_directory = 'GAN_4D'
# save_interval = 5000
# weight_of_reco_kicks_in_at = 0
# weight_of_reco_maxes_at = 0 # if weight_of_reco_maxes_at > 0:

# working_directory = '/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/TRAINING/'
# training_directory = '/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/DATA/'
# transformer_directory = '/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/TRANSFORMERS/'
# pre_trained_directory = '/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/PRE_TRAIN/'
# training_name = 'relu*.npy'
# testing_name = 'test_relu*.npy'
# saving_directory = 'GAN_4D'
# save_interval = 25000
# weight_of_reco_kicks_in_at = 0
# weight_of_reco_maxes_at = 0 # if weight_of_reco_maxes_at > 0:


working_directory = 'TRAINING/'
training_directory = '/hdfs/user/am13743/THESIS/DATA/'
transformer_directory = '/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/TRANSFORMERS/'
pre_trained_directory = '/hdfs/user/am13743/THESIS/PRE_TRAIN/'
training_name = 'relu*.npy'
testing_name = 'test_relu*.npy'
saving_directory = 'GAN_4D_just_relu'
save_interval = 25000
weight_of_reco_kicks_in_at = 0
weight_of_reco_maxes_at = 0 # if weight_of_reco_maxes_at > 0:
os.environ["CUDA_VISIBLE_DEVICES"]="1"


calculate_ROC = True
test_boosting = False

batch_size = 50

# G_architecture = [1000,1000]
# D_architecture = [1000,1000]

# A
G_architecture = [1000,1000,250,50]
D_architecture = [1000,1000,250,50]


D_architecture_aux = [32, 64]

weight_of_vanilla_loss = 10.
weight_of_reco_loss = 1.


list_of_training_files = glob.glob('%s%s'%(training_directory,training_name))

# try:
# 	files_to_remove = glob.glob('%s%s/CORRELATIONS/Correlations_*.png'%(working_directory,saving_directory))
# 	for file_i in files_to_remove:
# 		os.remove(file_i)
# except:
# 	print('/CORRELATIONS/ already clean')
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


optimizer = tf.keras.optimizers.Adam(lr=0.0004, beta_1=0.5, decay=0, amsgrad=True)

kernel_initializer_choice='random_uniform'
bias_initializer_choice='random_uniform'

##############################################################################################################
# Build Generative model ...
input_noise = Input(shape=(1,100))
auxiliary_inputs = Input(shape=(1,4))
charge_input = Input(shape=(1,1))
H_theta = Input(shape=(1,1))
H_pt_theta = Input(shape=(1,1))

initial_state = Concatenate()([input_noise,auxiliary_inputs,charge_input,H_theta,H_pt_theta])

H = Dense(int(G_architecture[0]),kernel_initializer=kernel_initializer_choice,bias_initializer=bias_initializer_choice)(initial_state)
H = LeakyReLU(alpha=0.2)(H)
H = BatchNormalization(momentum=0.8)(H)

for layer in G_architecture[1:]:

	H = Dense(int(layer),kernel_initializer=kernel_initializer_choice,bias_initializer=bias_initializer_choice)(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = BatchNormalization(momentum=0.8)(H)

H = Dense(6,kernel_initializer=kernel_initializer_choice,bias_initializer=bias_initializer_choice)(H)

H = Reshape((1,6))(H)

H_r = split_tensor(0, H)
H_r = ReLU(max_value=1.)(H_r)
H_r = Reshape((1,1))(H_r)

H_z = split_tensor(2, H)
H_z = ReLU(max_value=1.)(H_z)
H_z = Reshape((1,1))(H_z)

H_pt = split_tensor(3, H)
H_pt = ReLU(max_value=1.)(H_pt)
H_pt = Reshape((1,1))(H_pt)

H_pz = split_tensor(5, H)
H_pz = ReLU(max_value=1.)(H_pz)
H_pz = Reshape((1,1))(H_pz)

g_output = Concatenate(axis=-1)([H_r, H_theta, H_z, H_pt, H_pt_theta, H_pz])

g_output = Concatenate()([charge_input, g_output])

generator = Model(inputs=[input_noise,auxiliary_inputs,charge_input,H_theta,H_pt_theta], outputs=[g_output])
##############################################################################################################

##############################################################################################################
# Build Discriminator r model ...
d_input = Input(shape=(1,7))
H = split_tensor(1, d_input)
H = Flatten()(H)
for layer in D_architecture_aux:
	H = Dense(int(layer))(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = Dropout(0.2)(H)
d_output_aux = Dense(1, activation='relu',kernel_initializer=kernel_initializer_choice,bias_initializer=bias_initializer_choice)(H)
discriminator_aux_r = Model(d_input, d_output_aux)
discriminator_aux_r.load_weights('/%s/D_AUX_%s_WEIGHTS.h5'%(pre_trained_directory,'r'))

# Build Discriminator z model ...
d_input = Input(shape=(1,7))
H = split_tensor(3, d_input)
H = Flatten()(H)
for layer in D_architecture_aux:
	H = Dense(int(layer))(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = Dropout(0.2)(H)
d_output_aux = Dense(1, activation='relu')(H)
discriminator_aux_z = Model(d_input, d_output_aux)
discriminator_aux_z.load_weights('/%s/D_AUX_%s_WEIGHTS.h5'%(pre_trained_directory,'z'))

# Build Discriminator pt model ...
d_input = Input(shape=(1,7))
H = split_tensor(4, d_input)
H = Flatten()(H)
for layer in D_architecture_aux:
	H = Dense(int(layer))(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = Dropout(0.2)(H)
d_output_aux = Dense(1, activation='relu')(H)
discriminator_aux_pt = Model(d_input, d_output_aux)
discriminator_aux_pt.load_weights('/%s/D_AUX_%s_WEIGHTS.h5'%(pre_trained_directory,'pt'))

# Build Discriminator pz model ...
d_input = Input(shape=(1,7))
H = split_tensor(6, d_input)
H = Flatten()(H)
for layer in D_architecture_aux:
	H = Dense(int(layer))(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = Dropout(0.2)(H)
d_output_aux = Dense(1, activation='relu')(H)
discriminator_aux_pz = Model(d_input, d_output_aux)
discriminator_aux_pz.load_weights('/%s/D_AUX_%s_WEIGHTS.h5'%(pre_trained_directory,'pz'))
#############################################################################################################

##############################################################################################################
# Build Discriminator model ...
d_input = Input(shape=(1,7))

H = Flatten()(d_input)

for layer in D_architecture:

	H = Dense(int(layer),kernel_initializer=kernel_initializer_choice,bias_initializer=bias_initializer_choice)(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = Dropout(0.2)(H)
d_output = Dense(1, activation='sigmoid')(H)

discriminator = Model(d_input, [d_output])
##############################################################################################################


@tf.function
def train_step(images, weight_of_reco_loss_inner):

	images, aux_values_r, aux_values_z, aux_values_pt, aux_values_pz = images[:,:,:7], images[:,:,7], images[:,:,8], images[:,:,9], images[:,:,10]

	noise = tf.random.normal([batch_size, 1, 100])
	aux = tf.math.abs(tf.random.normal([batch_size, 1, 4]))
	charge_gan = tf.math.sign(tf.random.normal([batch_size, 1, 1]))
	theta_gan = tf.random.uniform([batch_size, 1, 1])*0.97+0.015
	theta_pt_gan = tf.random.uniform([batch_size, 1, 1])*0.97+0.015

	generated_images = generator([noise, aux, charge_gan, theta_gan, theta_pt_gan])

	in_values = tf.concat([generated_images, images],0)
	labels_D_0 = tf.zeros((batch_size, 1)) 
	labels_D_1 = tf.ones((batch_size, 1))
	labels_D = tf.concat([labels_D_0, labels_D_1],0)

	with tf.GradientTape(persistent=True) as disc_tape:
		out_values_choice = discriminator(in_values, training=True)
		disc_loss = tf.keras.losses.binary_crossentropy(tf.squeeze(labels_D),tf.squeeze(out_values_choice))

	noise_stacked = tf.random.normal((batch_size, 1, 100), 0, 1)
	aux_stacked = tf.math.abs(tf.random.normal([batch_size, 1, 4]))
	charge_stacked = tf.math.sign(tf.random.normal([batch_size, 1, 1]))
	labels_stacked = tf.ones((batch_size, 1))
	theta_stacked = tf.random.uniform([batch_size, 1, 1])*0.97+0.015
	theta_pt_stacked = tf.random.uniform([batch_size, 1, 1])*0.97+0.015

	with tf.GradientTape(persistent=True) as gen_tape:
		fake_images2 = generator([noise_stacked, aux_stacked, charge_stacked, theta_stacked, theta_pt_stacked], training=True)
		stacked_output_choice = discriminator(fake_images2)
		gen_loss_GAN = _loss_generator(tf.squeeze(labels_stacked),tf.squeeze(stacked_output_choice))

		stacked_output_reco_r = discriminator_aux_r(fake_images2)
		gen_loss_reco_r = tf.keras.losses.mean_squared_error(tf.squeeze(aux_stacked[:,:,0]),tf.squeeze(stacked_output_reco_r))

		stacked_output_reco_z = discriminator_aux_z(fake_images2)
		gen_loss_reco_z = tf.keras.losses.mean_squared_error(tf.squeeze(aux_stacked[:,:,1]),tf.squeeze(stacked_output_reco_z))

		stacked_output_reco_pt = discriminator_aux_pt(fake_images2)
		gen_loss_reco_pt = tf.keras.losses.mean_squared_error(tf.squeeze(aux_stacked[:,:,2]),tf.squeeze(stacked_output_reco_pt))

		stacked_output_reco_pz = discriminator_aux_pz(fake_images2)
		gen_loss_reco_pz = tf.keras.losses.mean_squared_error(tf.squeeze(aux_stacked[:,:,3]),tf.squeeze(stacked_output_reco_pz))

		gen_loss_reco = (gen_loss_reco_r+gen_loss_reco_z+gen_loss_reco_pt+gen_loss_reco_pz)

		gen_loss = gen_loss_reco*weight_of_reco_loss + gen_loss_GAN*weight_of_vanilla_loss

	grad_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
	optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))

	grad_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
	optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))

	return gen_loss, disc_loss, gen_loss_reco, gen_loss_GAN


start = time.time()

iteration = -1

loss_list = np.empty((0,5))

axis_titles = ['r', 'theta', 'StartZ', 'Pt', 'theta_pt', 'Pz']

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
					plt.xlabel(axis_titles[i])
					plt.ylabel(axis_titles[j])
			plt.subplots_adjust(wspace=0.3, hspace=0.3)
			plt.savefig('%s%s/CORRELATIONS_TRAIN.png'%(working_directory,saving_directory),bbox_inches='tight')
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

			if weight_of_reco_maxes_at > 0:
				if iteration < weight_of_reco_kicks_in_at:
					weight_of_reco_loss_i = tf.convert_to_tensor(0.)
				elif iteration > weight_of_reco_maxes_at:
					weight_of_reco_loss_i = tf.convert_to_tensor(weight_of_reco_loss)
				else:
					weight_of_reco_loss_i = tf.convert_to_tensor((weight_of_reco_loss/(weight_of_reco_maxes_at-weight_of_reco_kicks_in_at))*(iteration-weight_of_reco_kicks_in_at))
			else:
				weight_of_reco_loss_i = tf.convert_to_tensor(weight_of_reco_loss)

			gen_loss_np, disc_loss_np, gen_loss_np_reco, gen_loss_np_GAN = train_step(images_for_batch, weight_of_reco_loss_i)

			# loss_list = np.append(loss_list, [[iteration, gen_loss_np, disc_loss_np, gen_loss_np_reco, gen_loss_np_GAN]], axis=0)

			if iteration % save_interval == 0 and iteration > 0:

				t1 = time.time()

				total = t1-t0

				training_time += total

				print('Saving at iteration %d...'%iteration)

				# list_of_losses = ['Gen Loss', 'Dis Loss', 'D_aux r', 'D_aux z', 'D_aux Pt', 'D_aux Pz', 'Gen RECO', 'Gen GAN']

				# plt.figure(figsize=(16, 8))
				# plt.subplot(2,3,1)
				# plt.plot(loss_list[:,0], loss_list[:,1])
				# plt.ylabel('Gen Loss')
				# plt.subplot(2,3,2)
				# plt.plot(loss_list[:,0], loss_list[:,2])
				# plt.ylabel('Disc Loss')
				# plt.subplot(2,3,4)
				# plt.plot(loss_list[:,0], loss_list[:,3])
				# plt.ylabel('Gen Loss Reco')
				# plt.subplot(2,3,5)
				# plt.plot(loss_list[:,0], loss_list[:,4])
				# plt.ylabel('Gen Loss GAN')
				# plt.subplots_adjust(wspace=0.3, hspace=0.3)
				# plt.savefig('%s%s/LOSSES.png'%(working_directory,saving_directory),bbox_inches='tight')
				# plt.close('all')

				noise_size = 100000
				
				if iteration == 0: noise_size = 10
					
				charge_gan = np.random.choice([-1,1],size=(noise_size,1,1),p=[1-0.5,0.5],replace=True)
				aux_gan = np.abs(np.random.normal(0,1,size=(noise_size,4)))
				gen_noise = np.random.normal(0, 1, (int(noise_size), 100))
				theta_gan = np.random.uniform(low=0., high=1., size=(int(noise_size), 1, 1))*0.97+0.015
				theta_pt_gan = np.random.uniform(low=0., high=1., size=(int(noise_size), 1, 1))*0.97+0.015
				images = generator.predict([np.expand_dims(gen_noise,1), np.expand_dims(aux_gan,1), charge_gan, theta_gan, theta_pt_gan])

				images = np.squeeze(images)

				# Remove pdg info
				images = images[:,1:]
				samples = X_train[:,0,1:-5].copy()

				plt.figure(figsize=(5*4, 3*4))
				subplot=0
				for i in range(0, 6):
					for j in range(i+1, 6):
						subplot += 1
						plt.subplot(3,5,subplot)
						if subplot == 3: plt.title(iteration)
						plt.hist2d(images[:noise_size,i], images[:noise_size,j], bins=50,range=[[0,1],[0,1]], norm=LogNorm(), cmap=cmp_root)
						plt.xlabel(axis_titles[i])
						plt.ylabel(axis_titles[j])
				plt.subplots_adjust(wspace=0.3, hspace=0.3)
				# plt.savefig('%s%s/CORRELATIONS/Correlations_%d.png'%(working_directory,saving_directory,iteration),bbox_inches='tight')
				plt.savefig('%s%s/CORRELATIONS.png'%(working_directory,saving_directory),bbox_inches='tight')
				plt.close('all')

				plt.figure(figsize=(3*4, 2*4))
				subplot=0
				for i in range(0, 6):
					subplot += 1
					plt.subplot(2,3,subplot)
					if subplot == 2: plt.title(iteration)
					plt.hist([samples[:noise_size,i], images[:noise_size,i]], bins=50,range=[0,1], label=['Train','GEN'],histtype='step')
					plt.yscale('log')
					plt.xlabel(axis_titles[i])
					if axis_titles[i] == 'StartZ': plt.legend()
				plt.subplots_adjust(wspace=0.3, hspace=0.3)
				plt.savefig('%s%s/VALUES.png'%(working_directory,saving_directory),bbox_inches='tight')
				plt.close('all')

				if test_boosting == True:

					aux_gan[:,3] = aux_gan[:,3]*2.5
					images_wide = np.squeeze(generator.predict([np.expand_dims(gen_noise,1), np.expand_dims(aux_gan,1), charge_gan, theta_gan, theta_pt_gan]))
					images_wide = images_wide[:,1:]

					plt.figure(figsize=(5*4, 3*4))
					subplot=0
					for i in range(0, 6):
						for j in range(i+1, 6):
							subplot += 1
							plt.subplot(3,5,subplot)
							if subplot == 3: plt.title(iteration)
							plt.hist2d(images_wide[:noise_size,i], images_wide[:noise_size,j], bins=50,range=[[0,1],[0,1]], norm=LogNorm(), cmap=cmp_root)
							plt.xlabel(axis_titles[i])
							plt.ylabel(axis_titles[j])
					plt.subplots_adjust(wspace=0.3, hspace=0.3)
					plt.savefig('%s%s/CORRELATIONS_2_5.png'%(working_directory,saving_directory),bbox_inches='tight')
					# plt.savefig('%s%s/CORRELATIONS/Correlations_2_5_%d.png'%(working_directory,saving_directory,iteration),bbox_inches='tight')
					plt.close('all')

				if iteration > 0 and calculate_ROC == True:

					generator.save('%s%s/generator.h5'%(working_directory,saving_directory))
					discriminator.save('%s%s/discriminator.h5'%(working_directory,saving_directory))
					discriminator.save_weights('%s%s/discriminator_weights.h5'%(working_directory,saving_directory))

					###################################################

					clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)

					random_indicies = np.random.choice(list_for_np_choice, size=(noise_size), replace=False)
					X_train_sample = samples[random_indicies]

					bdt_train_size = int(np.shape(images)[0]/2)

					real_training_data = np.squeeze(X_train_sample[:bdt_train_size])

					real_test_data = np.squeeze(X_train_sample[bdt_train_size:])

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
					plt.legend(loc='upper right')
					plt.savefig('%s%s/BDT_out.png'%(working_directory,saving_directory), bbox_inches='tight')
					plt.close('all')

					ROC_AUC_SCORE_curr = roc_auc_score(np.append(np.ones(np.shape(out_real[:,1])),np.zeros(np.shape(out_fake[:,1]))),np.append(out_real[:,1],out_fake[:,1]))

					ROC_AUC_SCORE_list = np.append(ROC_AUC_SCORE_list, [[iteration, ROC_AUC_SCORE_curr, training_time]], axis=0)


					if ROC_AUC_SCORE_list[-1][1] < best_ROC_AUC:
						print('Saving best ROC_AUC.')
						generator.save('%s%s/Generator_best_ROC_AUC.h5'%(working_directory,saving_directory))
						discriminator.save('%s%s/Discriminator_best_ROC_AUC.h5'%(working_directory,saving_directory))
						# discriminator.save_weights('%s%s/Discriminator_best_ROC_AUC_weights.h5'%(working_directory,saving_directory))
						best_ROC_AUC = ROC_AUC_SCORE_list[-1][1]
						shutil.copy('%s%s/CORRELATIONS.png'%(working_directory,saving_directory), '%s%s/BEST_ROC_AUC_Correlations.png'%(working_directory,saving_directory))

					plt.figure(figsize=(8,4))
					plt.title('ROC_AUC_SCORE_list best: %.4f at %d'%(best_ROC_AUC,ROC_AUC_SCORE_list[np.where(ROC_AUC_SCORE_list==best_ROC_AUC)[0][0]][0]))
					plt.plot(ROC_AUC_SCORE_list[:,0],ROC_AUC_SCORE_list[:,1])
					plt.axhline(y=best_ROC_AUC,c='k',linestyle='--')
					plt.axvline(x=ROC_AUC_SCORE_list[np.where(ROC_AUC_SCORE_list==best_ROC_AUC)[0][0]][0],c='k',linestyle='--')
					plt.savefig('%s%s/ROC_progress.png'%(working_directory,saving_directory),bbox_inches='tight')
					plt.close('all')

					np.save('%s%s/FoM_ROC_AUC_SCORE_list'%(working_directory,saving_directory),ROC_AUC_SCORE_list)

				print('Saving complete.')
				t0 = time.time()
				



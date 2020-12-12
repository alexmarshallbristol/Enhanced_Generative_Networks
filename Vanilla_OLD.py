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

def _loss_generator(y_true, y_pred):
	y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
	out = -(K.log(y_pred))
	return K.mean(out, axis=-1)

def split_tensor(index, x):
    return Lambda(lambda x : x[:,:,index])(x)

print(tf.__version__)

# working_directory = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/TRAINING/'
# training_directory = '/Users/am13743/Aux_GAN_thesis/TRAINING/DATA/'
# transformer_directory = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/TRANSFORMERS/'
# training_name = 'smear*.npy'
# saving_directory = 'Vanilla_OLD'
# save_interval = 15
# min_max_GAN_paper = np.load('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/MIN_MAXES/min_max_GAN_paper.npy')
# min_max_smear = np.load('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/MIN_MAXES/min_max_smear.npy')
# min_max_ptparam = np.load('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/MIN_MAXES/min_max_ptparam.npy')


working_directory = '/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/TRAINING/'
training_directory = '/mnt/storage/scratch/am13743/AUX_GAN_THESIS/DATA/'
transformer_directory = '/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/TRANSFORMERS/'
training_name = 'smear*.npy'
testing_name = 'smear*.npy'
saving_directory = 'Vanilla_OLD'
save_interval = 25000
min_max_GAN_paper = np.load('/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/MIN_MAXES/min_max_GAN_paper.npy')
min_max_smear = np.load('/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/MIN_MAXES/min_max_smear.npy')
min_max_ptparam = np.load('/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/MIN_MAXES/min_max_ptparam.npy')


# working_directory = 'TRAINING/'
# training_directory = '/hdfs/user/am13743/THESIS/DATA/'
# transformer_directory = '/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/TRANSFORMERS/'
# pre_trained_directory = '/hdfs/user/am13743/THESIS/PRE_TRAIN/'
# training_name = 'smear*.npy'
# saving_directory = 'Vanilla_OLD'
# save_interval = 25000
# min_max_GAN_paper = np.load('/hdfs/user/am13743/THESIS/MIN_MAXES/min_max_GAN_paper.npy')
# min_max_smear = np.load('/hdfs/user/am13743/THESIS/MIN_MAXES/min_max_smear.npy')
# min_max_ptparam = np.load('/hdfs/user/am13743/THESIS/MIN_MAXES/min_max_ptparam.npy')
# os.environ["CUDA_VISIBLE_DEVICES"]="3"



def post_process_scaling(input_array, min_max):
	input_array[:,0] = (((input_array[:,0]+0.97)/1.94)*(min_max[0][1] - min_max[0][0])+ min_max[0][0])
	input_array[:,1] = (((input_array[:,1]+0.97)/1.94)*(min_max[1][1] - min_max[1][0])+ min_max[1][0])
	input_array[:,2] = (((input_array[:,2]+1.)/1.97)*(min_max[2][1] - min_max[2][0])+ min_max[2][0])
	input_array[:,3] = (((input_array[:,3]+0.97)/1.94)*(min_max[3][1] - min_max[3][0])+ min_max[3][0])
	input_array[:,4] = (((input_array[:,4]+0.97)/1.94)*(min_max[4][1] - min_max[4][0])+ min_max[4][0])
	input_array[:,5] = (((input_array[:,5]+0.97)/1.94)*(min_max[5][1] - min_max[5][0])+ min_max[5][0])
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

def pre_process_scaling(input_array, min_max):
	for index in [0,1,3,4,5]:
		range_i = min_max[index][1] - min_max[index][0]
		input_array[:,index] = ((input_array[:,index] - min_max[index][0])/range_i) * 1.94 - 0.97
	for index in [2]:
		range_i = min_max[index][1] - min_max[index][0]
		input_array[:,index] = ((input_array[:,index] - min_max[index][0])/range_i) * 1.97 - 1
	return input_array


calculate_ROC = True

batch_size = 50
virtual_batch_size = 50

G_architecture = [1000,1000,250,50]
D_architecture = [1000,1000,250,50]


list_of_training_files = glob.glob('%s%s'%(training_directory,training_name))

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
# Build Generative model ...
input_noise = Input(shape=(1,100))
charge_input = Input(shape=(1,1))

initial_state = Concatenate()([input_noise,charge_input])

H = Dense(int(G_architecture[0]))(initial_state)
H = LeakyReLU(alpha=0.2)(H)
H = BatchNormalization(momentum=0.8)(H)

for layer in G_architecture[1:]:

	H = Dense(int(layer))(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = BatchNormalization(momentum=0.8)(H)

H = Dense(6,activation='tanh')(H)

g_output = Reshape((1,6))(H)

g_output = Concatenate()([charge_input, g_output])

generator = Model(inputs=[input_noise,charge_input], outputs=[g_output])
##############################################################################################################

##############################################################################################################
# Build Discriminator model ...
d_input = Input(shape=(1,7))

H = Flatten()(d_input)

for layer in D_architecture:
	H = Dense(int(layer))(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = Dropout(0.2)(H)

d_output = Dense(1, activation='sigmoid')(H)

discriminator = Model(d_input, [d_output])
##############################################################################################################

gen_optimizer = tf.keras.optimizers.Adam(lr=0.0004, beta_1=0.5, decay=0, amsgrad=True)
disc_optimizer = tf.keras.optimizers.Adam(lr=0.0004, beta_1=0.5, decay=0, amsgrad=True)

@tf.function
def train_step(images):

	images, aux_values_BIN = images[:,:,:7], images[:,:,7:]

	noise = tf.random.normal([batch_size, 1, 100])
	charge_gan = tf.math.sign(tf.random.normal([batch_size, 1, 1]))

	generated_images = generator([noise, charge_gan])

	in_values = tf.concat([generated_images, images],0)
	labels_D_0 = tf.zeros((batch_size, 1)) 
	labels_D_1 = tf.ones((batch_size, 1))
	labels_D = tf.concat([labels_D_0, labels_D_1],0)

	with tf.GradientTape() as disc_tape:

		out_values = discriminator(in_values, training=True)
		disc_loss = tf.keras.losses.binary_crossentropy(labels_D,out_values)
		disc_loss = tf.math.reduce_mean(disc_loss)

	noise_stacked = tf.random.normal((batch_size, 1, 100), 0, 1)
	charge_stacked = tf.math.sign(tf.random.normal([batch_size, 1, 1]))
	labels_stacked = tf.ones((batch_size, 1))

	with tf.GradientTape() as gen_tape:

		fake_images2 = generator([noise_stacked, charge_stacked], training=True)
		stacked_output = discriminator(fake_images2)

		gen_loss = _loss_generator(labels_stacked,stacked_output)
		gen_loss = tf.math.reduce_mean(gen_loss)

	grad_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
	grad_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

	gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))
	disc_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))

	return gen_loss, disc_loss


start = time.time()

iteration = -1

loss_list = np.empty((0,3))

axis_titles_train = ['r', 'theta', 'z', 'pt', 'pt_theta', 'pz']
axis_titles_boxcox = ['r - BOXCOX', 'theta - QT', 'z - BOXCOX', 'pt - BOXCOX', 'pt_theta - QT', 'pz - BOXCOX']

ROC_AUC_SCORE_list = np.empty((0,3))
ROC_AUC_SCORE_list = np.append(ROC_AUC_SCORE_list, [[0, 1, 0]], axis=0)
best_ROC_AUC = 1

training_time = 0

t0 = time.time()

for epoch in range(int(1E30)):

	for file in list_of_training_files:

		batch_size = virtual_batch_size

		print('Loading initial training file:',file,'...')

		X_train = np.load(file)

		X_train = np.take(X_train,np.random.permutation(X_train.shape[0]),axis=0,out=X_train)

		print('Train images shape -',np.shape(X_train))

		list_for_np_choice = np.arange(np.shape(X_train)[0])

		X_train = np.expand_dims(X_train,1).astype("float32")

		train_dataset = (
			tf.data.Dataset.from_tensor_slices(X_train).batch(batch_size,drop_remainder=True).repeat(1)
		)

		for images_for_batch in train_dataset:

			if iteration % 250 == 0: print('Iteration:',iteration)

			if iteration > 50000 and iteration % 1000 == 0: virtual_batch_size += 1

			iteration += 1

			gen_loss_np, disc_loss_np = train_step(images_for_batch)

			loss_list = np.append(loss_list, [[iteration, gen_loss_np, disc_loss_np]], axis=0)

			if iteration % save_interval == 0:

				t1 = time.time()

				total = t1-t0

				training_time += total

				print('Saving at iteration %d...'%iteration)

				if iteration < 25000:
					plt.figure(figsize=(8, 8))
					plt.subplot(2,2,1)
					plt.plot(loss_list[:,0], loss_list[:,1])
					plt.ylabel('Gen Loss')
					plt.subplot(2,2,2)
					plt.plot(loss_list[:,0], loss_list[:,2])
					plt.ylabel('Disc Loss')
					plt.subplot(2,2,3)
					plt.plot(loss_list[:,0], loss_list[:,1])
					plt.ylabel('Gen Loss')
					plt.yscale('log')
					plt.subplot(2,2,4)
					plt.plot(loss_list[:,0], loss_list[:,2])
					plt.ylabel('Disc Loss')
					plt.yscale('log')
					plt.subplots_adjust(wspace=0.3, hspace=0.3)
					plt.savefig('%s%s/LOSSES.png'%(working_directory,saving_directory),bbox_inches='tight')
					plt.close('all')

				noise_size = 100000
		
				if iteration == 0: noise_size = 10
					
				charge_gan = np.random.choice([-1,1],size=(noise_size,1,1),p=[1-0.5,0.5],replace=True)
				gen_noise = np.random.normal(0, 1, (int(noise_size), 100))
				images = np.squeeze(generator.predict([np.expand_dims(gen_noise,1), charge_gan]))
				# Remove pdg info
				images = images[:,1:]

				samples = X_train[:,0,1:-5].copy()

				plt.figure(figsize=(3*4, 2*4))
				subplot=0
				for i in range(0, 6):
					subplot += 1
					plt.subplot(2,3,subplot)
					if subplot == 2: plt.title(iteration)
					plt.hist([samples[:noise_size,i], images[:noise_size,i]], bins=50,range=[-1,1], label=['Train','GEN'],histtype='step')
					plt.yscale('log')
					plt.xlabel(axis_titles_train[i])
					if axis_titles_train[i] == 'z': plt.legend()
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
						plt.hist2d(images[:noise_size,i], images[:noise_size,j], bins=50,range=[[-1,1],[-1,1]], norm=LogNorm(), cmap=cmp_root)
						plt.xlabel(axis_titles_train[i])
						plt.ylabel(axis_titles_train[j])
				plt.subplots_adjust(wspace=0.3, hspace=0.3)
				# plt.savefig('%s%s/CORRELATIONS/Correlations_%d.png'%(working_directory,saving_directory,iteration),bbox_inches='tight')
				plt.savefig('%s%s/CORRELATIONS.png'%(working_directory,saving_directory),bbox_inches='tight')
				plt.close('all')


				if iteration > 0 and calculate_ROC == True:

					try:
						generator.save('%s%s/generator.h5'%(working_directory,saving_directory))
						discriminator.save('%s%s/discriminator.h5'%(working_directory,saving_directory))
						discriminator.save_weights('%s%s/discriminator_weights.h5'%(working_directory,saving_directory))

						###################################################

						clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)

						random_indicies = np.random.choice(list_for_np_choice, size=(noise_size), replace=False)
						X_train_sample = samples[random_indicies].copy()

						X_train_sample = post_process_scaling(X_train_sample,min_max_smear)
						X_train_sample = pxpy_to_ptparam(X_train_sample)
						X_train_sample = pre_process_scaling(X_train_sample,min_max_ptparam)
						X_train_sample = (X_train_sample + 1.)/2.

						images = post_process_scaling(images,min_max_smear)
						images = pxpy_to_ptparam(images)
						images = pre_process_scaling(images,min_max_ptparam)
						images = (images + 1.)/2.

						plt.figure(figsize=(5*4, 3*4))
						subplot=0
						for i in range(0, 6):
							for j in range(i+1, 6):
								subplot += 1
								plt.subplot(3,5,subplot)
								if subplot == 3: plt.title(iteration)
								plt.hist2d(images[:noise_size,i], images[:noise_size,j], bins=50,range=[[-1,1],[-1,1]], norm=LogNorm(), cmap=cmp_root)
								plt.xlabel(axis_titles_train[i])
								plt.ylabel(axis_titles_train[j])
						plt.subplots_adjust(wspace=0.3, hspace=0.3)
						plt.savefig('%s%s/BDT_data_example.png'%(working_directory,saving_directory),bbox_inches='tight')
						plt.close('all')

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
						# plt.savefig('%s%s/bdt/BDT_P_out_%d.png'%(working_directory,saving_directory,cnt), bbox_inches='tight')
						plt.savefig('%s%s/BDT_out.png'%(working_directory,saving_directory), bbox_inches='tight')
						plt.close('all')

						ROC_AUC_SCORE_curr = roc_auc_score(np.append(np.ones(np.shape(out_real[:,1])),np.zeros(np.shape(out_fake[:,1]))),np.append(out_real[:,1],out_fake[:,1]))

						ROC_AUC_SCORE_list = np.append(ROC_AUC_SCORE_list, [[iteration, ROC_AUC_SCORE_curr, training_time]], axis=0)


						if ROC_AUC_SCORE_list[-1][1] < best_ROC_AUC:
							print('Saving best ROC_AUC.')
							generator.save('%s%s/Generator_best_ROC_AUC.h5'%(working_directory,saving_directory))
							discriminator.save('%s%s/Discriminator_best_ROC_AUC.h5'%(working_directory,saving_directory))
							discriminator.save_weights('%s%s/Discriminator_best_ROC_AUC_weights.h5'%(working_directory,saving_directory))
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
					except:
						print('Roc failed')

				print('Saving complete.')
				t0 = time.time()
				


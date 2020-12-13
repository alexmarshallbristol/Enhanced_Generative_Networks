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
import random

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
plt.rcParams['savefig.dpi'] = 250
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rcParams.update({'font.size': 15})

def _loss_generator(y_true, y_pred):
	y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
	out = -(K.log(y_pred))
	return K.mean(out, axis=-1)

def split_tensor(index, x):
    return Lambda(lambda x : x[:,:,index])(x)

print(tf.__version__)


pre_trained_directory = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/PRE_TRAIN/'

D_architecture_aux = [32, 64]

generator = load_model('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/BLUE_CRYSTAL_RESULTS/GAN_4D/generator.h5')


##############################################################################################################
# Build Discriminator r model ...
d_input = Input(shape=(1,7))
H = split_tensor(1, d_input)
H = Flatten()(H)
for layer in D_architecture_aux:
	H = Dense(int(layer))(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = Dropout(0.2)(H)
d_output_aux = Dense(1, activation='relu')(H)
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


def post_process_scaling(input_array, min_max):
	input_array[:,0] = (((input_array[:,0]+0.97)/1.94)*(min_max[0][1] - min_max[0][0])+ min_max[0][0])
	input_array[:,1] = (((input_array[:,1]+0.97)/1.94)*(min_max[1][1] - min_max[1][0])+ min_max[1][0])
	input_array[:,2] = (((input_array[:,2]+1.)/1.97)*(min_max[2][1] - min_max[2][0])+ min_max[2][0])
	input_array[:,3] = (((input_array[:,3]+0.97)/1.94)*(min_max[3][1] - min_max[3][0])+ min_max[3][0])
	input_array[:,4] = (((input_array[:,4]+0.97)/1.94)*(min_max[4][1] - min_max[4][0])+ min_max[4][0])
	input_array[:,5] = (((input_array[:,5]+0.97)/1.94)*(min_max[5][1] - min_max[5][0])+ min_max[5][0])
	return input_array

def pre_process_scaling(input_array, min_max):
	for index in [0,1,3,4,5]:
		range_i = min_max[index][1] - min_max[index][0]
		input_array[:,index] = ((input_array[:,index] - min_max[index][0])/range_i) * 1.94 - 0.97
	for index in [2]:
		range_i = min_max[index][1] - min_max[index][0]
		input_array[:,index] = ((input_array[:,index] - min_max[index][0])/range_i) * 1.97 - 1
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

def ptparam_to_pxpy(input_array):
	x = np.expand_dims(input_array[:,0]*np.sin(input_array[:,1]),1)
	y = np.expand_dims(input_array[:,0]*np.cos(input_array[:,1]),1)
	z = np.expand_dims(input_array[:,2],1)
	px = np.expand_dims(input_array[:,3]*np.sin(input_array[:,4]),1)
	py = np.expand_dims(input_array[:,3]*np.cos(input_array[:,4]),1)
	pz = np.expand_dims(input_array[:,5],1)
	input_array = np.concatenate((x,y,z,px,py,pz),axis=1)
	return input_array


def pre_process_THETAS(input_array, min_max):
	for index in [0,1,3,4,5]:
		range_i = min_max[index][1] - min_max[index][0]
		input_array[:,index] = ((input_array[:,index] - min_max[index][0])/range_i) * 1.94 - 0.97
	for index in [2]:
		range_i = min_max[index][1] - min_max[index][0]
		input_array[:,index] = ((input_array[:,index] - min_max[index][0])/range_i) * 1.97 - 1
	return input_array

min_max_GAN_paper = np.load('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/MIN_MAXES/min_max_GAN_paper.npy')
min_max_smear = np.load('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/MIN_MAXES/min_max_smear.npy')
min_max_ptparam = np.load('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/MIN_MAXES/min_max_ptparam.npy')


points = np.asarray([[0,5,-7000,0,0,100],
						[5,0,-7050,-1,2,50],
							[2.5,-2.,-7050,0.5,-0.9,250],
								[-5,-4,2-7000,-1.6,-1,150],
									[1,-6,2-6850,0.4,0.2,20]
									])
original_points = points.copy()

points_theta = np.expand_dims(np.arctan2(points[:,0],points[:,1]),1)
points_pt_theta = np.expand_dims(np.arctan2(points[:,3],points[:,4]),1)

index_thetas = 1
range_i = min_max_ptparam[index_thetas][1] - min_max_ptparam[index_thetas][0]
points_theta = ((points_theta - min_max_ptparam[index_thetas][0])/range_i) * 1.94 - 0.97

index_thetas = 4
range_i = min_max_ptparam[index_thetas][1] - min_max_ptparam[index_thetas][0]
points_pt_theta = ((points_pt_theta - min_max_ptparam[index_thetas][0])/range_i) * 1.94 - 0.97



points = pxpy_to_ptparam(points)
points = pre_process_scaling(points, min_max_ptparam)

print(np.shape(points))
# add pdg
points = np.expand_dims(np.concatenate((np.ones((np.shape(points)[0],1)),points),axis=1),1)
print(np.shape(points))

print(points)

aux = np.concatenate((discriminator_aux_r.predict(points),discriminator_aux_z.predict(points),discriminator_aux_pt.predict(points),discriminator_aux_pz.predict(points)),axis=1)

print(np.shape(aux), np.shape(points_theta), np.shape(points_pt_theta))

print(aux)
# quit()
new_sample_all = []

num_points = 100
angle_blur_width = 0.03

for point_index in range(np.shape(points)[0]):

	aux_tile= np.tile(aux[point_index], (num_points,1))

	points_theta_tile = np.tile(points_theta[point_index], (num_points,1))

	points_pt_theta_tile = np.tile(points_pt_theta[point_index], (num_points,1))

	charge_gan = np.random.choice([-1,1],size=(num_points,1,1),p=[1-0.5,0.5],replace=True)
	gen_noise = np.random.normal(0, 1, (num_points, 100))


	points_theta_tile = points_theta_tile*1/0.97
	points_theta_tile += np.random.normal(0,angle_blur_width,size=(np.shape(points_theta_tile)))
	points_theta_tile[np.where(points_theta_tile>1)] += -2
	points_theta_tile[np.where(points_theta_tile<-1)] += 2
	points_theta_tile = points_theta_tile*0.97

	points_pt_theta_tile = points_pt_theta_tile*1/0.97
	points_pt_theta_tile += np.random.normal(0,angle_blur_width,size=(np.shape(points_pt_theta_tile)))
	points_pt_theta_tile[np.where(points_pt_theta_tile>1)] += -2
	points_pt_theta_tile[np.where(points_pt_theta_tile<-1)] += 2
	points_pt_theta_tile = points_pt_theta_tile*0.97

	new_sample = np.squeeze(generator.predict([np.expand_dims(gen_noise,1), np.expand_dims(aux_tile,1), charge_gan, np.expand_dims(points_theta_tile,1), np.expand_dims(points_pt_theta_tile,1)]))[:,1:]

	new_sample = post_process_scaling(new_sample,min_max_ptparam)
	new_sample = ptparam_to_pxpy(new_sample)

	new_sample_all.append(new_sample)

new_sample_all = np.asarray(new_sample_all)









points = original_points.copy()



colours = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']

axislabels_indexes=['x (cm)','y (cm)','z (cm)','$P_x$ (GeV/c)','$P_y$ (GeV/c)','$P_z$ (GeV/c)']

current = np.load("REAL_DATA_16.npy")[:1000000]


subplot = 0
indexes = [0,1,2,3,4,5]
labels_indexes=['x','y','z','$P_x$','$P_y$','$P_z$']
axislabels_indexes=['x (cm)','y (cm)','z (cm)','$P_x$ (GeV/c)','$P_y$ (GeV/c)','$P_z$ (GeV/c)']

ticks_i=[[-5,0,5],[-5,0,5],[-7000,-6800,-6600],[-3,0,3],[-3,0,3],[100,200,300]]

min_values_posx = np.empty(6)
max_values_posx = np.empty(6)

min_values_posx[0] = -10
max_values_posx[0] = 10

min_values_posx[1] = -10
max_values_posx[1] = 10

min_values_posx[2] = np.amin(current[:,2])
max_values_posx[2] = -6590

min_values_posx[3] = -4
max_values_posx[3] = 4

min_values_posx[4] = -4
max_values_posx[4] = 4

min_values_posx[5] = np.amin(current[:,5])
max_values_posx[5] = np.amax(current[:,5])

# np.save('min_max_plotting',np.concatenate((np.expand_dims(min_values_posx,1),np.expand_dims(max_values_posx,1)),axis=1))

# quit()

max_bin_value = 1

fig = plt.figure(figsize=(24,24))

subplot = 1
for j in range(0, 6):
	subplot = j + 1
	subplot += 6 * (j + 1)
	add_6 = 0
	for i in range(j+1, 6):
		if add_6 != 0:
			subplot += 6
		print(i,j,subplot)

		ax = fig.add_subplot(6,6,subplot)
		# plt.hist2d(real_data[:,indexes[j]],real_data[:,indexes[i]],bins=75,norm=LogNorm(),cmap=cmp_root,range=[[min_values_posx[j],max_values_posx[j]],[min_values_posx[i],max_values_posx[i]]], vmin=1,vmax=52777.0)
		# # plt.hist2d(real_data[:,indexes[j]],real_data[:,indexes[i]],bins=75,norm=LogNorm(),cmap=plt.cm.Blues_r,range=[[min_values_posx[j],max_values_posx[j]],[min_values_posx[i],max_values_posx[i]]],vmin=1,vmax=14471275.0)
		# hist = np.histogram2d(real_data[:,indexes[j]],real_data[:,indexes[i]],bins=75,range=[[min_values_posx[j],max_values_posx[j]],[min_values_posx[i],max_values_posx[i]]])
		# # plt.grid(axis='both', which='major',color='k',linestyle='--',alpha=0.3)

		plt.hist2d(current[:,indexes[j]], current[:,indexes[i]], bins=75, norm=LogNorm(), cmap=cmp_root, alpha=0.3,range=[[min_values_posx[j],max_values_posx[j]],[min_values_posx[i],max_values_posx[i]]])
		hist = np.histogram2d(current[:,indexes[j]],current[:,indexes[i]],bins=75,range=[[min_values_posx[j],max_values_posx[j]],[min_values_posx[i],max_values_posx[i]]])

		for point_index in range(0, np.shape(points)[0]):
			new_sample_all_i = new_sample_all[point_index]
			plt.scatter(new_sample_all_i[:,j], new_sample_all_i[:,i],c=colours[point_index],alpha=0.6)
		for point_index in range(0, np.shape(points)[0]):
			plt.scatter(points[point_index][j], points[point_index][i],c=colours[point_index],alpha=1.,marker='s',s=40,edgecolors='k')


		if np.amax(hist[0]) > max_bin_value:
			max_bin_value = np.amax(hist[0])
			print(max_bin_value)

		plt.xticks(ticks_i[j],ticks_i[j])
		plt.yticks(ticks_i[i],ticks_i[i])

		if subplot in [7,13,19,25]:
			plt.xticks([])
			# plt.gca().yaxis.tick_right()
			# plt.gca().xaxis.tick_top()
			# continue
			plt.ylabel(axislabels_indexes[i],fontsize=30)
		elif subplot == 31:
			plt.ylabel(axislabels_indexes[i],fontsize=30)
			plt.xlabel(axislabels_indexes[j],fontsize=30)
			continue
		elif subplot in [32,33,34,35,36]:
			plt.yticks([])
			plt.xlabel(axislabels_indexes[j],fontsize=30)
		else:
			plt.xticks([])
			plt.yticks([])

		# ax.text(0.9,0.1, '%s'%labels_indexes[j],horizontalalignment='center',verticalalignment='center',transform=ax.transAxes,size=30)
		# ax.text(0.1,0.9, '%s'%labels_indexes[i],horizontalalignment='center',verticalalignment='center',transform=ax.transAxes,size=30)

		add_6 += 1

subplot = 1 - 7
for j in range(0, 6):
	# print(i)
	subplot += 7
	print(j,subplot)

	ax = fig.add_subplot(6,6,subplot)

	# plt.hist([real_data[:,indexes[j]],gen_raw[:,indexes[j]]],bins=75,range=[min_values_posx[j],max_values_posx[j]],histtype='step',linewidth=3,color=['#094F9A','#BA1219'])
	# plt.hist(real_data[:,indexes[j]],bins=75,range=[-1,1],histtype='step',linewidth=3,color=['#4772FF','#FF5959'])
	plt.hist(current[:,indexes[j]],bins=75,range=[min_values_posx[j],max_values_posx[j]],histtype='step',linewidth=3,color=colours_raw_root[2])

	for point_index in range(0, np.shape(points)[0]):
		new_sample_all_i = new_sample_all[point_index]
		plt.hist(new_sample_all_i[:,j],bins=75,range=[min_values_posx[j],max_values_posx[j]],histtype='step',linewidth=3,color=colours[point_index],alpha=0.6)

	if j in [0,1,2,3,4,5]:
		plt.yscale('log')

	ax.text(0.95,0.9, 'Log-scale',horizontalalignment='right',verticalalignment='center',transform=ax.transAxes,size=18)
	if subplot not in [1,36]:
		plt.xticks([])
		plt.yticks([])
	elif subplot == 36:
		plt.yticks([])
		plt.xlabel(axislabels_indexes[j],fontsize=30)
	else:
		# plt.yticks([])
		plt.xticks([])
		plt.ylabel(axislabels_indexes[j],fontsize=30)
	# ax.set_xticks([])
	# ax.set_yticks([])
	ax.tick_params(direction='in') 

print(max_bin_value)



ax2 = fig.add_axes([0.6, 0.6, 0.25, 0.25])
plt.hist2d(np.sqrt(current[:,3]**2+current[:,4]**2+current[:,5]**2),np.sqrt(current[:,3]**2+current[:,4]**2),bins=75,norm=LogNorm(),cmap=cmp_root,range=[[0,400],[0,5]], vmin=1,vmax=52777.0, alpha=0.3)


cax = plt.axes([0.9, 0.35, 0.01, 0.3])
cbar = plt.colorbar(cax=cax)
# cbar.ax.set_ylabel('Full Simulation', rotation=270,fontsize=30,labelpad=40)
# cbar.ax.set_xlabel('a.u.')
cbar.ax.tick_params(labelsize=30)


hist = np.histogram2d(np.sqrt(current[:,3]**2+current[:,4]**2+current[:,5]**2),np.sqrt(current[:,3]**2+current[:,4]**2),bins=75,range=[[0,400],[0,5]])
if np.amax(hist[0]) > max_bin_value:
	max_bin_value = np.amax(hist[0])
	print(max_bin_value)

ax2 = fig.add_axes([0.6, 0.6, 0.25, 0.25])

for point_index in range(0, np.shape(points)[0]):
	new_sample_all_i = new_sample_all[point_index]
	plt.scatter(np.sqrt(new_sample_all_i[:,3]**2+new_sample_all_i[:,4]**2+new_sample_all_i[:,5]**2), np.sqrt(new_sample_all_i[:,3]**2+new_sample_all_i[:,4]**2),c=colours[point_index],alpha=0.6)
	# plt.scatter(new_sample_all_i[:,j], new_sample_all_i[:,i],c=colours[point_index],alpha=0.6)
for point_index in range(0, np.shape(points)[0]):
	# plt.scatter(points[point_index][j], points[point_index][i],c=colours[point_index],alpha=1.,marker='s',s=40,edgecolors='k')
	plt.scatter(np.sqrt(points[point_index][3]**2+points[point_index][4]**2+points[point_index][5]**2), np.sqrt(points[point_index][3]**2+points[point_index][4]**2),c=colours[point_index],alpha=1.,marker='s',s=40,edgecolors='k')





print('max',max_bin_value)

plt.xlabel('Momentum (GeV/c)',fontsize=30)
plt.ylabel('Transverse Momentum (GeV/c)',fontsize=30)

plt.subplots_adjust(bottom=0.1, right=0.85, left=0.075, top=0.9, wspace = 0,hspace = 0)

plt.savefig('THESIS_PLOTS/EVO.pdf',bbox_inches='tight')
plt.close('all')








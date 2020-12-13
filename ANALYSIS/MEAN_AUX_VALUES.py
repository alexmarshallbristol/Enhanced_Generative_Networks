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

def post_process_scaling(input_array, min_max):
	input_array[:,0] = (((input_array[:,0]+0.97)/1.94)*(min_max[0][1] - min_max[0][0])+ min_max[0][0])
	input_array[:,1] = (((input_array[:,1]+0.97)/1.94)*(min_max[1][1] - min_max[1][0])+ min_max[1][0])
	input_array[:,2] = (((input_array[:,2]+1.)/1.97)*(min_max[2][1] - min_max[2][0])+ min_max[2][0])
	input_array[:,3] = (((input_array[:,3]+0.97)/1.94)*(min_max[3][1] - min_max[3][0])+ min_max[3][0])
	input_array[:,4] = (((input_array[:,4]+0.97)/1.94)*(min_max[4][1] - min_max[4][0])+ min_max[4][0])
	input_array[:,5] = (((input_array[:,5]+0.97)/1.94)*(min_max[5][1] - min_max[5][0])+ min_max[5][0])
	return input_array


data = np.load('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/DATA/relu_9.npy')

# files = glob.glob('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/DATA/relu_*.npy')

# data = np.empty((0,12))
# for file in files:
# 	# print(np.shape(np.load(file)))
# 	# quit()
# 	data = np.concatenate((data, np.load(file)),axis=0)

data[:,1:7] = (data[:,1:7]*2.)-1.
min_max_ptparam = np.load('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/MIN_MAXES/min_max_ptparam.npy')
data[:,1:7] = post_process_scaling(data[:,1:7], min_max_ptparam)






subplot = 0
indexes = [0,1,2,3,4,5]
labels_indexes=['x','y','z','$P_x$','$P_y$','$P_z$']
axislabels_indexes=['r (cm)',r'$\theta_r$ (rads)','z (cm)','$P_t$ (GeV/c)',r'$\theta_p$ (rads)','$P_z$ (GeV/c)']

# ticks_i=[[-5,0,5],[-5,0,5],[-7000,-6800,-6600],[-3,0,3],[-3,0,3],[100,200,300]]

min_values_posx = np.empty(6)
max_values_posx = np.empty(6)

min_values_posx[0] = np.amin(data[:,1])
max_values_posx[0] = np.amax(data[:,1])

min_values_posx[1] = -math.pi
max_values_posx[1] = math.pi

min_values_posx[2] = np.amin(data[:,3])
max_values_posx[2] = np.amax(data[:,3])

min_values_posx[3] = 0
max_values_posx[3] = np.amax(np.abs(data[:,4]))

min_values_posx[4] = -math.pi
max_values_posx[4] = math.pi

min_values_posx[5] = np.amin(data[:,6])
max_values_posx[5] = np.amax(data[:,6])

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

		# plt.hist2d(current[:,indexes[j]], current[:,indexes[i]], bins=75, norm=LogNorm(), cmap=cmp_root,range=[[min_values_posx[j],max_values_posx[j]],[min_values_posx[i],max_values_posx[i]]],vmin=1,vmax=910691)
		# hist = np.histogram2d(current[:,indexes[j]],current[:,indexes[i]],bins=75,range=[[min_values_posx[j],max_values_posx[j]],[min_values_posx[i],max_values_posx[i]]])
		
		hist = np.histogram2d(data[:,j+1], data[:,i+1], bins=50, weights=data[:,-1], range=[[min_values_posx[j],max_values_posx[j]],[min_values_posx[i],max_values_posx[i]]])[0]
		hist[np.where(hist==0)] = None
		hist = hist / np.histogram2d(data[:,j+1], data[:,i+1], bins=50, range=[[min_values_posx[j],max_values_posx[j]],[min_values_posx[i],max_values_posx[i]]])[0]
		plt.imshow(np.fliplr(hist).T,aspect='auto',cmap=cmp_root,extent=[min_values_posx[j],max_values_posx[j],min_values_posx[i],max_values_posx[i]],vmin=0, vmax=5)

		# print('np.amax(hist)', np.amax(hist))

		

		# plt.xticks(ticks_i[j],ticks_i[j])
		# plt.yticks(ticks_i[i],ticks_i[i])

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
# for j in range(0, 6):
# 	# print(i)
# 	subplot += 7
# 	print(j,subplot)

# 	ax = fig.add_subplot(6,6,subplot)

# 	# plt.hist([real_data[:,indexes[j]],gen_raw[:,indexes[j]]],bins=75,range=[min_values_posx[j],max_values_posx[j]],histtype='step',linewidth=3,color=['#094F9A','#BA1219'])
# 	# plt.hist(real_data[:,indexes[j]],bins=75,range=[-1,1],histtype='step',linewidth=3,color=['#4772FF','#FF5959'])
# 	plt.hist(current[:,indexes[j]],bins=75,range=[min_values_posx[j],max_values_posx[j]],histtype='step',linewidth=3,color=colours_raw_root[2])

# 	for point_index in range(0, np.shape(points)[0]):
# 		new_sample_all_i = new_sample_all[point_index]
# 		plt.hist(new_sample_all_i[:,j],bins=75,range=[min_values_posx[j],max_values_posx[j]],histtype='step',linewidth=3,color=colours[point_index],alpha=0.6)

# 	if j in [0,1,2,3,4,5]:
# 		plt.yscale('log')

# 	ax.text(0.95,0.9, 'Log-scale',horizontalalignment='right',verticalalignment='center',transform=ax.transAxes,size=18)
# 	if subplot not in [1,36]:
# 		plt.xticks([])
# 		plt.yticks([])
# 	elif subplot == 36:
# 		plt.yticks([])
# 		plt.xlabel(axislabels_indexes[j],fontsize=30)
# 	else:
# 		# plt.yticks([])
# 		plt.xticks([])
# 		plt.ylabel(axislabels_indexes[j],fontsize=30)
# 	# ax.set_xticks([])
# 	# ax.set_yticks([])
# 	ax.tick_params(direction='in') 

print(max_bin_value)



# ax2 = fig.add_axes([0.6, 0.6, 0.25, 0.25])
# plt.hist2d(np.sqrt(current[:,3]**2+current[:,4]**2+current[:,5]**2),np.sqrt(current[:,3]**2+current[:,4]**2),bins=75,norm=LogNorm(),cmap=cmp_root,range=[[0,400],[0,5]], vmin=1,vmax=52777.0, alpha=0.3)


cax = plt.axes([0.76, 0.35, 0.01, 0.3])
cbar = plt.colorbar(cax=cax)
cbar.ax.set_ylabel('Mean Auxiliary Value', rotation=270,fontsize=30,labelpad=40)
# cbar.ax.set_xlabel('a.u.')
cbar.ax.tick_params(labelsize=30)





print('max',max_bin_value)

# plt.xlabel('$P$ (GeV/c)',fontsize=30)
# plt.ylabel('$P_t$ (GeV/c)',fontsize=30)

plt.subplots_adjust(bottom=0.1, right=0.85, left=0.075, top=0.9, wspace = 0,hspace = 0)

plt.savefig('THESIS_PLOTS/MEAN_AUX_VALUES.pdf',bbox_inches='tight')
# plt.savefig('mean.png',bbox_inches='tight')
plt.close('all')



quit()


axis_titles = ['r', 'theta', 'z', 'pt', 'pt_theta', 'pz']
for aux_id_i in range(0,5):
	plt.figure(figsize=(4*5,4*3))
	subplot = 0
	for i in range(0, 6):
		for j in range(i+1, 6):
			subplot += 1
			plt.subplot(3,5,subplot)
			i_max = 1
			i_min = -1
			j_max = 1
			j_min = -1
			hist = np.histogram2d(current[:,i+1], current[:,j+1], bins=50, weights=current[:,aux_id_i+7], range=[[i_min,i_max],[j_min,j_max]])[0]
			hist[np.where(hist==0)] = None
			hist = hist / np.histogram2d(current[:,i+1], current[:,j+1], bins=50, range=[[i_min,i_max],[j_min,j_max]])[0]
			plt.imshow(np.fliplr(hist).T,aspect='auto',cmap=cmp_root,extent=[i_min,i_max,j_min,j_max])
			plt.xlabel(axis_titles[i])
			plt.ylabel(axis_titles[j])
	plt.subplots_adjust(hspace=0.3,wspace=0.3)
	plt.savefig('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/DATA/PTPARAM_aux_values_%d.pdf'%aux_id_i,bbox_inches='tight')
	plt.close('all')
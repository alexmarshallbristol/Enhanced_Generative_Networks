import numpy as np
import matplotlib as mpl
mpl.use('TkAgg') 
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pickle import load, dump
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
import glob
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

training_directory = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/DATA/'
transformer_directory = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/TRANSFORMERS/'
training_name = 'data*.npy'

trans_1 = load(open('%strans_1.pkl'%transformer_directory, 'rb'))
trans_2 = load(open('%strans_2.pkl'%transformer_directory, 'rb'))
trans_3 = load(open('%strans_3.pkl'%transformer_directory, 'rb'))
trans_4 = load(open('%strans_4.pkl'%transformer_directory, 'rb'))
trans_5 = load(open('%strans_5.pkl'%transformer_directory, 'rb'))
trans_6 = load(open('%strans_6.pkl'%transformer_directory, 'rb'))

def inverse_transform_qt_boxcox(input_array):
	input_array[:,0] = np.squeeze(trans_1.inverse_transform(np.expand_dims(input_array[:,0],1)*7.))
	input_array[:,1] = np.squeeze(trans_2.inverse_transform(np.expand_dims(input_array[:,1],1)*7.))
	input_array[:,2] = np.squeeze(trans_3.inverse_transform(np.expand_dims(input_array[:,2],1)*7.))
	input_array[:,3] = np.squeeze(trans_4.inverse_transform(np.expand_dims(input_array[:,3],1)*7.))
	input_array[:,4] = np.squeeze(trans_5.inverse_transform(np.expand_dims(input_array[:,4],1)*7.))
	input_array[:,5] = np.squeeze(trans_6.inverse_transform(np.expand_dims(input_array[:,5],1)*7.))
	input_array = ((input_array - 0.1) * 2.4) - 1.
	return input_array

list_of_training_files = glob.glob('%s%s'%(training_directory,training_name))

data = np.load(list_of_training_files[0])

print(np.shape(data))

data[:,1:7] = inverse_transform_qt_boxcox(data[:,1:7])

axis_titles = ['r','theta','z','pt','pt_theta','pz']

plt.figure(figsize=(5*4, 3*4))
subplot=0
for i in range(0, 6):
	for j in range(i+1, 6):
		subplot += 1
		plt.subplot(3,5,subplot)
		plt.hist2d(data[:,i+1], data[:,j+1], bins=50,range=[[-1,1],[-1,1]], cmap=cmp_root)
		plt.xlabel(axis_titles[i])
		plt.ylabel(axis_titles[j])
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig('%s/BOOSTING_EXAMPLE/SAMPLE.png'%(training_directory),bbox_inches='tight')
plt.close('all')

# Artificially boost the sample based on 4D inverse local density

boost_weights = data[:,-1]
boost_weights = np.squeeze(boost_weights/np.sum(boost_weights))
list_for_np_choice = np.arange(np.shape(boost_weights)[0])
random_indicies_order = np.random.choice(list_for_np_choice, size=np.shape(data)[0], p=boost_weights, replace=True) 
data = data[random_indicies_order]

plt.figure(figsize=(5*4, 3*4))
subplot=0
for i in range(0, 6):
	for j in range(i+1, 6):
		subplot += 1
		plt.subplot(3,5,subplot)
		plt.hist2d(data[:,i+1], data[:,j+1], bins=50,range=[[-1,1],[-1,1]], cmap=cmp_root)
		plt.xlabel(axis_titles[i])
		plt.ylabel(axis_titles[j])
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig('%s/BOOSTING_EXAMPLE/SAMPLE_BOOSTED.png'%(training_directory),bbox_inches='tight')
plt.close('all')










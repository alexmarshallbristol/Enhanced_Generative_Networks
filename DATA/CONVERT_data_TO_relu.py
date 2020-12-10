import numpy as np
import glob
import matplotlib as mpl
# mpl.use('TkAgg') 
# mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math
from pickle import load, dump
from sklearn.neighbors import NearestNeighbors
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

transformer_directory = '/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/TRANSFORMERS/'
min_max_location = '/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/MIN_MAXES/'
data_location = '/mnt/storage/scratch/am13743/AUX_GAN_THESIS/THESIS_ITERATION/DATA/'
data_name = 'data*.npy'
test_data_name = 'test_data*.npy'

# transformer_directory = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/TRANSFORMERS/'
# min_max_location = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/MIN_MAXES/'
# data_location = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/DATA/'
# data_name = 'data*.npy'
# test_data_name = 'test_data*.npy'

files = glob.glob('%s%s'%(data_location,data_name))

min_max_GAN_paper = np.load('%smin_max_GAN_paper.npy'%min_max_location)
min_max_smear = np.load('%smin_max_smear.npy'%min_max_location)
min_max_ptparam = np.load('%smin_max_ptparam.npy'%min_max_location)

indexes = {'charge':0, 'r':1, 'theta':2, 'z':3, 'pt':4, 'pt_theta':5, 'pz':6, 'r_aux':7, 'z_aux':8, 'pt_aux':9, 'pz_aux':10, '4D_aux':11}

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

def transform_to_relu(input_array):
	input_array = (input_array + 1. ) /2.
	return input_array

print('data files...')
for file_id, file in enumerate(files):
	print(file_id)
	current = np.load(file)
	current[:,1:7] = inverse_transform_qt_boxcox(current[:,1:7])
	current[:,1:7] = transform_to_relu(current[:,1:7])
	np.save('%srelu_%d.npy'%(data_location,file_id), current)

print('test_data files...')
files = glob.glob('%s%s'%(data_location,test_data_name))
for file_id, file in enumerate(files):
	print(file_id)
	current = np.load(file)
	current[:,1:7] = inverse_transform_qt_boxcox(current[:,1:7])
	current[:,1:7] = transform_to_relu(current[:,1:7])
	np.save('%stest_relu_%d.npy'%(data_location,file_id), current)

	
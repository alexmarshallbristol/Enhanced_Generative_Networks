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


files = glob.glob('/Volumes/Mac-Toshiba/PhD/Muon_Shield_Opt_GAN_sample/Hard_map2/hardmap_mu_data*.npy')

indexes = {'charge':0, 'x':1, 'y':2, 'z':3, 'px':4, 'py':5, 'pz':6}
indexes_ptparam = {'charge':0, 'r':1, 'theta':2, 'z':3, 'pt':4, 'pt_theta':5, 'pz':6, 'r_aux':7, 'z_aux':8, 'pt_aux':9, 'pz_aux':10, '4D_aux':11}


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

def transform_qt_boxcox(input_array):
	input_array = ((input_array + 1.)/2.4) + 0.1
	input_array[:,0] = np.squeeze(trans_1.transform(np.expand_dims(input_array[:,0],1)))/7.
	input_array[:,1] = np.squeeze(trans_2.transform(np.expand_dims(input_array[:,1],1)))/7.
	input_array[:,2] = np.squeeze(trans_3.transform(np.expand_dims(input_array[:,2],1)))/7.
	input_array[:,3] = np.squeeze(trans_4.transform(np.expand_dims(input_array[:,3],1)))/7.
	input_array[:,4] = np.squeeze(trans_5.transform(np.expand_dims(input_array[:,4],1)))/7.
	input_array[:,5] = np.squeeze(trans_6.transform(np.expand_dims(input_array[:,5],1)))/7.
	return input_array

def inverse_transform_qt_boxcox(input_array):
	input_array[:,0] = np.squeeze(trans_1.inverse_transform(np.expand_dims(input_array[:,0],1)*7.))
	input_array[:,1] = np.squeeze(trans_2.inverse_transform(np.expand_dims(input_array[:,1],1)*7.))
	input_array[:,2] = np.squeeze(trans_3.inverse_transform(np.expand_dims(input_array[:,2],1)*7.))
	input_array[:,3] = np.squeeze(trans_4.inverse_transform(np.expand_dims(input_array[:,3],1)*7.))
	input_array[:,4] = np.squeeze(trans_5.inverse_transform(np.expand_dims(input_array[:,4],1)*7.))
	input_array[:,5] = np.squeeze(trans_6.inverse_transform(np.expand_dims(input_array[:,5],1)*7.))
	input_array = ((input_array - 0.1) * 2.4) - 1.
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



min_max_GAN_paper = np.load('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/MIN_MAXES/min_max_GAN_paper.npy')
min_max_smear = np.load('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/MIN_MAXES/min_max_smear.npy')
min_max_ptparam = np.load('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/MIN_MAXES/min_max_ptparam.npy')

training_file = True
plotting_example = True

transformer_directory = '/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/TRANSFORMERS/'

file_id = -2

for file in files:

	current = np.load(file)

	if np.shape(current)[0] == 1000000:

		if np.amin(current[:,4]) > -1:

			file_id += 1
			print('file_id:',file_id,'train bool:',training_file,'plot bool:',plotting_example)

			columns_to_remove = [0, 8, 9, 10, 11] # Remove weights and previous auxiliary values

			current = np.delete(current, columns_to_remove, axis=1)

			# print(np.shape(current)) # (1000000, 7)

			# Un-transform the GAN paper x y transformation
			current[:,indexes['x']] = np.sign(current[:,indexes['x']])*current[:,indexes['x']]**2
			current[:,indexes['y']] = np.sign(current[:,indexes['y']])*current[:,indexes['y']]**2

			# Add empty columns to fill with AUX values
			current = np.concatenate((current, np.zeros((np.shape(current)[0],5))),axis=1)

			# Return to physical values
			current[:,1:7] = post_process_scaling(current[:,1:7], min_max_GAN_paper)

			# Smear x and y
			r = 5 + 0.8*np.random.normal(0,1,size=np.shape(current)[0])
			phi = np.random.uniform(low=0, high=math.pi*2, size=np.shape(current)[0])	
			current[:,indexes['x']] += r*np.cos(phi)
			current[:,indexes['y']] += r*np.sin(phi)

			# Convert to P_t parameterisation
			current[:,1:7] = pxpy_to_ptparam(current[:,1:7])

			# Return to normalized values
			current[:,1:7] = pre_process_scaling(current[:,1:7], min_max_ptparam)

			if training_file == True:

				print('Training file == True, so fitting NearestNeighbors...')

				current_trans = current.copy()
				for index in range(1, 7):
					if index in [2, 5]:
						trans = QuantileTransformer(n_quantiles=500, output_distribution='normal')
					else:
						trans = PowerTransformer(method='box-cox')
					current_trans[:,index] = ((current_trans[:,index] + 1.) / 2.4 ) + 0.1
					X = np.expand_dims(current_trans[:,index],1)
					trans.fit(X)
					dump(trans, open('%strans_%d.pkl'%(transformer_directory,index), 'wb'))

				trans_1 = load(open('%strans_1.pkl'%transformer_directory, 'rb'))
				trans_2 = load(open('%strans_2.pkl'%transformer_directory, 'rb'))
				trans_3 = load(open('%strans_3.pkl'%transformer_directory, 'rb'))
				trans_4 = load(open('%strans_4.pkl'%transformer_directory, 'rb'))
				trans_5 = load(open('%strans_5.pkl'%transformer_directory, 'rb'))
				trans_6 = load(open('%strans_6.pkl'%transformer_directory, 'rb'))

				num_points = 10000 # -1

				current = np.take(current,np.random.permutation(current.shape[0]),axis=0,out=current)
				columns_to_fit_to = [indexes_ptparam['r'],indexes_ptparam['z'],indexes_ptparam['pt'],indexes_ptparam['pz']]
				fit_to_this = current[:num_points,columns_to_fit_to]
				neighbors_4D_aux = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(fit_to_this)

				current = np.take(current,np.random.permutation(current.shape[0]),axis=0,out=current)
				columns_to_fit_to = [indexes_ptparam['r']]
				fit_to_this = current[:num_points,columns_to_fit_to]
				neighbors_r_aux = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(fit_to_this)

				current = np.take(current,np.random.permutation(current.shape[0]),axis=0,out=current)
				columns_to_fit_to = [indexes_ptparam['z']]
				fit_to_this = current[:num_points,columns_to_fit_to]
				neighbors_z_aux = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(fit_to_this)

				current = np.take(current,np.random.permutation(current.shape[0]),axis=0,out=current)
				columns_to_fit_to = [indexes_ptparam['pt']]
				fit_to_this = current[:num_points,columns_to_fit_to]
				neighbors_pt_aux = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(fit_to_this)

				current = np.take(current,np.random.permutation(current.shape[0]),axis=0,out=current)
				columns_to_fit_to = [indexes_ptparam['pz']]
				fit_to_this = current[:num_points,columns_to_fit_to]
				neighbors_pz_aux = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(fit_to_this)

				training_file = False

				print('Fitting complete.')

				continue

			current = np.take(current,np.random.permutation(current.shape[0]),axis=0,out=current)
			columns_to_query_to = [indexes_ptparam['r'],indexes_ptparam['z'],indexes_ptparam['pt'],indexes_ptparam['pz']]
			query_to_this = current[:,columns_to_query_to]
			distances, indices = neighbors_4D_aux.kneighbors(query_to_this)
			average_distances = np.mean(distances,axis=1)
			current[:,indexes_ptparam['4D_aux']] = average_distances
			current = current[current[:,indexes_ptparam['4D_aux']].argsort()]
			half_gauss = np.abs(np.random.normal(loc=0, scale=1, size=(np.shape(current[:,indexes_ptparam['4D_aux']])[0],1)))
			half_gauss = half_gauss[half_gauss[:,0].argsort()]
			current[:,indexes_ptparam['4D_aux']] = half_gauss[:,0]

			physical_indexes = ['r','z','pt','pz']
			auxiliary_indexes = ['r_aux','z_aux','pt_aux','pz_aux']
			NearestNeighbor_objects = [neighbors_r_aux, neighbors_z_aux, neighbors_pt_aux, neighbors_pz_aux]

			for index in range(0,4):
				current = np.take(current,np.random.permutation(current.shape[0]),axis=0,out=current)
				columns_to_query_to = [indexes_ptparam[physical_indexes[index]]]
				query_to_this = current[:,columns_to_query_to]
				distances, indices = NearestNeighbor_objects[index].kneighbors(query_to_this)
				average_distances = np.mean(distances,axis=1)
				current[:,indexes_ptparam[auxiliary_indexes[index]]] = average_distances
				current = current[current[:,indexes_ptparam[auxiliary_indexes[index]]].argsort()]
				half_gauss = np.abs(np.random.normal(loc=0, scale=1, size=(np.shape(current[:,indexes_ptparam[auxiliary_indexes[index]]])[0],1)))
				half_gauss = half_gauss[half_gauss[:,0].argsort()]
				current[:,indexes_ptparam[auxiliary_indexes[index]]] = half_gauss[:,0]

			current = np.take(current,np.random.permutation(current.shape[0]),axis=0,out=current)

			if plotting_example == True:
				print('Plotting PTPARAM')
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

			# Perform QT and BOXCOX transforms
			current[:,1:7] = transform_qt_boxcox(current[:,1:7])

			if training_file == False:
				print('Saving file...')
				np.save('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/DATA/data_%d'%file_id, current)

			if plotting_example == True:
				print('Plotting QT Box-Cox')
				axis_titles = ['r - BOXCOX', 'theta - QT', 'z - BOXCOX', 'pt - BOXCOX', 'pt_theta - QT', 'pz - BOXCOX']
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
					plt.savefig('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/DATA/QTBOXCOX_aux_values_%d.pdf'%aux_id_i,bbox_inches='tight')
					plt.close('all')

				current[:,1:7] = inverse_transform_qt_boxcox(current[:,1:7])
				current[:,1:7] = post_process_scaling(current[:,1:7], min_max_ptparam)
				current[:,1:7] = ptparam_to_pxpy(current[:,1:7])
				current[:,1:7] = pre_process_scaling(current[:,1:7], min_max_smear)

				print('Plotting Norm Physical')
				axis_titles = ['x', 'y', 'z', 'px', 'py', 'pz']
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
					plt.savefig('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/DATA/Norm_physical_aux_values_%d.pdf'%aux_id_i,bbox_inches='tight')
					plt.close('all')

				plotting_example = False


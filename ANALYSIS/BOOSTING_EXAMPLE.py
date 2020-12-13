import numpy as np
import random
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
import time
plt.rc('text', usetex=True)
plt.rcParams['savefig.dpi'] = 250
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rcParams.update({'font.size': 15})
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

colours_raw_root_norm = np.flip(np.divide(colours_raw_root,256.),axis=0)
cmp_root = mpl.colors.ListedColormap(colours_raw_root_norm)

# import glob

# training_files = glob.glob('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/DATA/relu*.npy')
# training_files2 = glob.glob('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/DATA/test_relu*.npy')

# gen_files = glob.glob('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/ANALYSIS/GEN_DATA/gen_raw*.npy')

# gen_files_boosted = glob.glob('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/ANALYSIS/GEN_DATA/gen_boost*.npy')

# min_max = np.load('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/MIN_MAXES/min_max_ptparam.npy')

# def post_process_scaling(input_array, min_max):
# 	input_array = (input_array * 2.) - 1.
# 	input_array[:,0] = (((input_array[:,0]+0.97)/1.94)*(min_max[0][1] - min_max[0][0])+ min_max[0][0])
# 	input_array[:,1] = (((input_array[:,1]+0.97)/1.94)*(min_max[1][1] - min_max[1][0])+ min_max[1][0])
# 	input_array[:,2] = (((input_array[:,2]+1.)/1.97)*(min_max[2][1] - min_max[2][0])+ min_max[2][0])
# 	input_array[:,3] = (((input_array[:,3]+0.97)/1.94)*(min_max[3][1] - min_max[3][0])+ min_max[3][0])
# 	input_array[:,4] = (((input_array[:,4]+0.97)/1.94)*(min_max[4][1] - min_max[4][0])+ min_max[4][0])
# 	input_array[:,5] = (((input_array[:,5]+0.97)/1.94)*(min_max[5][1] - min_max[5][0])+ min_max[5][0])
# 	return input_array
# def ptparam_to_pxpy(input_array):
# 	x = np.expand_dims(input_array[:,0]*np.sin(input_array[:,1]),1)
# 	y = np.expand_dims(input_array[:,0]*np.cos(input_array[:,1]),1)
# 	z = np.expand_dims(input_array[:,2],1)
# 	px = np.expand_dims(input_array[:,3]*np.sin(input_array[:,4]),1)
# 	py = np.expand_dims(input_array[:,3]*np.cos(input_array[:,4]),1)
# 	pz = np.expand_dims(input_array[:,5],1)
# 	input_array = np.concatenate((x,y,z,px,py,pz),axis=1)
# 	return input_array

# real_data = np.empty((0,6))

# for file in training_files:
# 	current = np.load(file)[:,1:7]
# 	current = post_process_scaling(current,min_max)
# 	current = ptparam_to_pxpy(current)

# 	real_data = np.append(real_data, current, axis=0)

# for file in training_files2:
# 	current = np.load(file)[:,1:7]
# 	current = post_process_scaling(current,min_max)
# 	current = ptparam_to_pxpy(current)

# 	real_data = np.append(real_data, current, axis=0)


# gen_raw = np.empty((0,6))

# for file in gen_files:
# 	current = np.load(file)
# 	gen_raw = np.append(gen_raw, current, axis=0)



# gen_boost = np.empty((0,6))

# for file in gen_files_boosted:
# 	current = np.load(file)
# 	gen_boost = np.append(gen_boost, current, axis=0)

# np.save("REAL_DATA_16",real_data)
# np.save("GEN_RAW_16",gen_raw)
# np.save("GEN_BOOST_16",gen_boost)

# quit()
real_data = np.load("REAL_DATA_16.npy")
gen_raw = np.load("GEN_RAW_16.npy")
gen_boost = np.load("GEN_BOOST_16.npy")

real_data = real_data[:np.shape(gen_raw)[0]]

plt.figure(figsize=(10,4))
ax = plt.subplot(1,2,1)
plt.hist2d(np.sqrt(real_data[:,3]**2+real_data[:,4]**2+real_data[:,5]**2),np.sqrt(real_data[:,3]**2+real_data[:,4]**2),bins=100,range=[[0,400],[0,6]],norm=LogNorm(),cmap=plt.cm.Blues_r)
plt.ylabel('Transverse Momentum (GeV)', fontsize=15)
plt.xlabel('Momentum (GeV)', fontsize=15)
plt.text(0.95, 0.95,'Full Simulation',
     horizontalalignment='right',
     verticalalignment='top',
     transform = ax.transAxes, fontsize=15)


ax = plt.subplot(1,2,2)
plt.hist2d(np.sqrt(gen_raw[:,3]**2+gen_raw[:,4]**2+gen_raw[:,5]**2),np.sqrt(gen_raw[:,3]**2+gen_raw[:,4]**2),bins=100,range=[[0,400],[0,6]],norm=LogNorm(),cmap=plt.cm.Reds_r)
plt.ylabel('Transverse Momentum (GeV)', fontsize=15)
plt.xlabel('Momentum (GeV)', fontsize=15)
plt.text(0.95, 0.95,'Enhanced GAN Generation',
     horizontalalignment='right',
     verticalalignment='top',
     transform = ax.transAxes, fontsize=15)


# plt.subplot(2,2,3)
# plt.hist([np.sqrt(gen_raw[:,3]**2+gen_raw[:,4]**2+gen_raw[:,5]**2),np.sqrt(real_data[:,3]**2+real_data[:,4]**2+real_data[:,5]**2)],bins=100,range=[0,400],label=['GAN Generation','Full Simulation'],histtype='step',linewidth=1.75)
# plt.yscale('log')
# plt.xlabel('Momentum (GeV)')

# plt.subplot(2,2,4)
# plt.hist([np.sqrt(gen_raw[:,3]**2+gen_raw[:,4]**2),np.sqrt(real_data[:,3]**2+real_data[:,4]**2)],bins=100,range=[0,6],label=['GAN Generation','Full Simulation'],histtype='step',linewidth=1.75)
# plt.yscale('log')
# plt.ylabel('Transverse Momentum (GeV)')
# plt.legend()

plt.subplots_adjust(hspace=0.25, wspace=0.25)
plt.savefig('THESIS_PLOTS/PPT.pdf',bbox_inches='tight')
plt.close('all')




print(np.shape(real_data), np.shape(gen_raw))

real_data = np.take(real_data,np.random.permutation(real_data.shape[0]),axis=0,out=real_data)

gen_raw = np.take(gen_raw,np.random.permutation(gen_raw.shape[0]),axis=0,out=gen_raw)


real_data = real_data[:np.shape(gen_raw)[0]]



max_gan = [np.max(gen_boost[:,0]),np.max(gen_boost[:,1]),np.max(gen_boost[:,2]),np.max(gen_boost[:,3]),np.max(gen_boost[:,4]),np.max(gen_boost[:,5])]
min_gan = [np.min(gen_boost[:,0]),np.min(gen_boost[:,1]),np.min(gen_boost[:,2]),np.min(gen_boost[:,3]),np.min(gen_boost[:,4]),np.min(gen_boost[:,5])]

max_thomas = [np.max(real_data[:,0]),np.max(real_data[:,1]),np.max(real_data[:,2]),np.max(real_data[:,3]),np.max(real_data[:,4]),np.max(real_data[:,5])]
min_thomas = [np.min(real_data[:,0]),np.min(real_data[:,1]),np.min(real_data[:,2]),np.min(real_data[:,3]),np.min(real_data[:,4]),np.min(real_data[:,5])]

max_values_posx = np.empty(6)
min_values_posx = np.empty(6)


for i in range(0, 6):
	max_values_posx[i] = np.max([max_gan[i],max_thomas[i]])
	min_values_posx[i] = np.min([min_gan[i],min_thomas[i]])


symmetric = [0,1,3,4]

for i in symmetric:
	values = [np.absolute(min_values_posx[i]),np.absolute(max_values_posx[i])]
	max_value = np.max(values)
	min_values_posx[i] = -1 * max_value
	max_values_posx[i] = 1 * max_value

# make x and y the same

min_values_posx[0] = min_values_posx[1]
max_values_posx[0] = max_values_posx[1]

min_values_posx[3] = min_values_posx[4]
max_values_posx[3] = max_values_posx[4]

min_values_posx[5] = 0







subplot = 0
indexes = [0,1,2,3,4,5]
labels_indexes=['x','y','z','$P_x$','$P_y$','$P_z$']
axislabels_indexes=['x (cm)','y (cm)','z (cm)','$P_x$ (GeV/c)','$P_y$ (GeV/c)','$P_z$ (GeV/c)']

ticks_i=[[-5,0,5],[-5,0,5],[-7000,-6800,-6600],[-5,0,5],[-5,0,5],[100,200,300]]


max_bin_value = 1


fig = plt.figure(figsize=(12,12))
for subplot in [2,3,6]:

		if subplot == 2:
			i = 3
			j = 4
		elif subplot == 3:
			i = 3
			j = 5
		elif subplot == 6:
			i = 4
			j = 5

		ax = fig.add_subplot(3,3,subplot)
		plt.hist2d(gen_raw[:,j],gen_raw[:,i],bins=75,norm=LogNorm(),cmap=plt.cm.Reds_r,range=[[min_values_posx[j],max_values_posx[j]],[min_values_posx[i],max_values_posx[i]]],vmin=1,vmax=833011)
		# plt.hist2d(gen_raw[:,i],gen_raw[:,j],bins=75,norm=LogNorm(),cmap=cmp_root_inv,range=[[min_values_posx[i],max_values_posx[i]],[min_values_posx[j],max_values_posx[j]]])

		hist = np.histogram2d(gen_raw[:,j],gen_raw[:,i],bins=75,range=[[min_values_posx[j],max_values_posx[j]],[min_values_posx[i],max_values_posx[i]]])

		if np.amax(hist[0]) > max_bin_value:
			max_bin_value = np.amax(hist[0])
			print(max_bin_value)

		# plt.xticks(ticks_i[j],ticks_i[j])
		# plt.yticks(ticks_i[i],ticks_i[i])

		# plt.tick_params(axis='y', which='both', labelsize=15)
		# plt.tick_params(axis='x', which='both', labelsize=15)

		if subplot in [2]:
			plt.yticks([])
			plt.gca().xaxis.tick_top()

			ax.set_xlabel(axislabels_indexes[j],fontsize=15)    
			ax.xaxis.set_label_position('top') 
		elif subplot== 3:
			plt.gca().xaxis.tick_top()
			plt.gca().yaxis.tick_right()
			ax.set_xlabel(axislabels_indexes[j],fontsize=15)    
			ax.xaxis.set_label_position('top')
			ax.set_ylabel(axislabels_indexes[i],fontsize=15)    
			ax.yaxis.set_label_position('right')
			continue
		elif subplot in [6]:
			plt.gca().yaxis.tick_right()
			plt.xticks([])
			ax.set_ylabel(axislabels_indexes[i],fontsize=15)    
			ax.yaxis.set_label_position('right')
		else:
			plt.xticks([])
			plt.yticks([])

		# ax.text(0.9,0.1, '%s'%labels_indexes[j],horizontalalignment='center',verticalalignment='center',transform=ax.transAxes,size=30)
		# ax.text(0.1,0.9, '%s'%labels_indexes[i],horizontalalignment='center',verticalalignment='center',transform=ax.transAxes,size=30)

cax = plt.axes([0.95, 0.4333333333+0.1333333333, 0.01, 0.3])
cbar = plt.colorbar(cax=cax)
cbar.ax.set_ylabel('GAN Generated', rotation=270,fontsize=15,labelpad=40)
cbar.ax.tick_params(labelsize=15)

for subplot in [4,7,8]:

		if subplot == 4:
			i = 3
			j = 4
		elif subplot == 7:
			i = 3
			j = 5
		elif subplot == 8:
			i = 4
			j = 5

		ax = fig.add_subplot(3,3,subplot)
		plt.hist2d(real_data[:,i],real_data[:,j],bins=75,norm=LogNorm(),cmap=plt.cm.Blues_r,range=[[min_values_posx[i],max_values_posx[i]],[min_values_posx[j],max_values_posx[j]]],vmin=1,vmax=833011)
		# plt.hist2d(real_data[:,i],real_data[:,j],bins=75,norm=LogNorm(),cmap=cmp_root,range=[[min_values_posx[i],max_values_posx[i]],[min_values_posx[j],max_values_posx[j]]])

		hist = np.histogram2d(real_data[:,i],real_data[:,j],bins=75,range=[[min_values_posx[i],max_values_posx[i]],[min_values_posx[j],max_values_posx[j]]])

		if np.amax(hist[0]) > max_bin_value:
			max_bin_value = np.amax(hist[0])
			print(max_bin_value)

		# plt.xticks(ticks_i[j],ticks_i[j])
		# plt.yticks(ticks_i[i],ticks_i[i])

		# plt.tick_params(axis='y', which='both', labelsize=15)
		# plt.tick_params(axis='x', which='both', labelsize=15)

		if subplot in [4]:
			plt.xticks([])
			# plt.gca().yaxis.tick_right()
			# plt.gca().xaxis.tick_top()
			# continue
			plt.ylabel(axislabels_indexes[j],fontsize=15)
		elif subplot == 7:
			plt.ylabel(axislabels_indexes[j],fontsize=15)
			plt.xlabel(axislabels_indexes[i],fontsize=15)
			continue
		elif subplot in [8]:
			plt.yticks([])
			plt.xlabel(axislabels_indexes[i],fontsize=15)
		else:
			plt.xticks([])
			plt.yticks([])

		# ax.text(0.9,0.1, '%s'%labels_indexes[j],horizontalalignment='center',verticalalignment='center',transform=ax.transAxes,size=30)
		# ax.text(0.1,0.9, '%s'%labels_indexes[i],horizontalalignment='center',verticalalignment='center',transform=ax.transAxes,size=30)

cax = plt.axes([0.95, 0.1333333333, 0.01, 0.3])
cbar = plt.colorbar(cax=cax)
cbar.ax.set_ylabel('Full Simulation', rotation=270,fontsize=15,labelpad=40)
# cbar.ax.set_xlabel('a.u.')
cbar.ax.tick_params(labelsize=15)

for subplot in [1,5,9]:
	# print(i)
	if subplot == 1:
		i = 3
	elif subplot == 5:
		i = 4
	elif subplot == 9:
		i = 5

	ax = fig.add_subplot(3,3,subplot)

	# plt.hist([real_data[:,indexes[j]],gen_raw[:,indexes[j]]],bins=75,range=[min_values_posx[j],max_values_posx[j]],histtype='step',linewidth=3,color=['#094F9A','#BA1219'])
	plt.hist([real_data[:,i],gen_raw[:,i]],bins=75,range=[min_values_posx[i],max_values_posx[i]],histtype='step',linewidth=3,color=['#4772FF','#FF5959'])

	plt.yscale('log')

	ax.text(0.95,0.9, 'Log-scale',horizontalalignment='right',verticalalignment='center',transform=ax.transAxes,size=18)
	plt.xticks([])
	plt.yticks([])
	# ax.set_xticks([])
	# ax.set_yticks([])
	ax.tick_params(direction='in') 



plt.subplots_adjust(bottom=0.1, right=0.85, left=0.075, top=0.9, wspace = 0,hspace = 0)

plt.savefig('THESIS_PLOTS/BOOSTING_EXAMPLE_1.pdf',bbox_inches='tight')
plt.close('all')





subplot = 0
indexes = [0,1,2,3,4,5]
labels_indexes=['x','y','z','$P_x$','$P_y$','$P_z$']
axislabels_indexes=['x (cm)','y (cm)','z (cm)','$P_x$ (GeV/c)','$P_y$ (GeV/c)','$P_z$ (GeV/c)']

ticks_i=[[-5,0,5],[-5,0,5],[-7000,-6800,-6600],[-5,0,5],[-5,0,5],[100,200,300]]


max_bin_value = 1


fig = plt.figure(figsize=(12,12))
for subplot in [2,3,6]:

		if subplot == 2:
			i = 3
			j = 4
		elif subplot == 3:
			i = 3
			j = 5
		elif subplot == 6:
			i = 4
			j = 5

		ax = fig.add_subplot(3,3,subplot)
		plt.hist2d(gen_boost[:,j],gen_boost[:,i],bins=75,norm=LogNorm(),cmap=plt.cm.Reds_r,range=[[min_values_posx[j],max_values_posx[j]],[min_values_posx[i],max_values_posx[i]]],vmin=1,vmax=833011)

		hist = np.histogram2d(gen_boost[:,j],gen_boost[:,i],bins=75,range=[[min_values_posx[j],max_values_posx[j]],[min_values_posx[i],max_values_posx[i]]])

		if np.amax(hist[0]) > max_bin_value:
			max_bin_value = np.amax(hist[0])
			print(max_bin_value)

		# plt.xticks(ticks_i[j],ticks_i[j])
		# plt.yticks(ticks_i[i],ticks_i[i])

		# plt.tick_params(axis='y', which='both', labelsize=15)
		# plt.tick_params(axis='x', which='both', labelsize=15)

		if subplot in [2]:
			plt.yticks([])
			plt.gca().xaxis.tick_top()

			ax.set_xlabel(axislabels_indexes[j],fontsize=15)    
			ax.xaxis.set_label_position('top') 
		elif subplot== 3:
			plt.gca().xaxis.tick_top()
			plt.gca().yaxis.tick_right()
			ax.set_xlabel(axislabels_indexes[j],fontsize=15)    
			ax.xaxis.set_label_position('top')
			ax.set_ylabel(axislabels_indexes[i],fontsize=15)    
			ax.yaxis.set_label_position('right')
			continue
		elif subplot in [6]:
			plt.gca().yaxis.tick_right()
			plt.xticks([])
			ax.set_ylabel(axislabels_indexes[i],fontsize=15)    
			ax.yaxis.set_label_position('right')
		else:
			plt.xticks([])
			plt.yticks([])

		# ax.text(0.9,0.1, '%s'%labels_indexes[j],horizontalalignment='center',verticalalignment='center',transform=ax.transAxes,size=30)
		# ax.text(0.1,0.9, '%s'%labels_indexes[i],horizontalalignment='center',verticalalignment='center',transform=ax.transAxes,size=30)

cax = plt.axes([0.95, 0.4333333333+0.1333333333, 0.01, 0.3])
cbar = plt.colorbar(cax=cax)
cbar.ax.set_ylabel('GAN Generated (Boosted)', rotation=270,fontsize=15,labelpad=40)
cbar.ax.tick_params(labelsize=15)

for subplot in [4,7,8]:

		if subplot == 4:
			i = 3
			j = 4
		elif subplot == 7:
			i = 3
			j = 5
		elif subplot == 8:
			i = 4
			j = 5

		ax = fig.add_subplot(3,3,subplot)
		plt.hist2d(real_data[:,i],real_data[:,j],bins=75,norm=LogNorm(),cmap=plt.cm.Blues_r,range=[[min_values_posx[i],max_values_posx[i]],[min_values_posx[j],max_values_posx[j]]],vmin=1,vmax=833011)

		hist = np.histogram2d(real_data[:,i],real_data[:,j],bins=75,range=[[min_values_posx[i],max_values_posx[i]],[min_values_posx[j],max_values_posx[j]]])

		if np.amax(hist[0]) > max_bin_value:
			max_bin_value = np.amax(hist[0])
			print(max_bin_value)

		# plt.xticks(ticks_i[j],ticks_i[j])
		# plt.yticks(ticks_i[i],ticks_i[i])

		# plt.tick_params(axis='y', which='both', labelsize=15)
		# plt.tick_params(axis='x', which='both', labelsize=15)

		if subplot in [4]:
			plt.xticks([])
			# plt.gca().yaxis.tick_right()
			# plt.gca().xaxis.tick_top()
			# continue
			plt.ylabel(axislabels_indexes[j],fontsize=15)
		elif subplot == 7:
			plt.ylabel(axislabels_indexes[j],fontsize=15)
			plt.xlabel(axislabels_indexes[i],fontsize=15)
			continue
		elif subplot in [8]:
			plt.yticks([])
			plt.xlabel(axislabels_indexes[i],fontsize=15)
		else:
			plt.xticks([])
			plt.yticks([])

		# ax.text(0.9,0.1, '%s'%labels_indexes[j],horizontalalignment='center',verticalalignment='center',transform=ax.transAxes,size=30)
		# ax.text(0.1,0.9, '%s'%labels_indexes[i],horizontalalignment='center',verticalalignment='center',transform=ax.transAxes,size=30)

cax = plt.axes([0.95, 0.1333333333, 0.01, 0.3])
cbar = plt.colorbar(cax=cax)
cbar.ax.set_ylabel('Full Simulation', rotation=270,fontsize=15,labelpad=40)
# cbar.ax.set_xlabel('a.u.')
cbar.ax.tick_params(labelsize=15)

for subplot in [1,5,9]:
	# print(i)
	if subplot == 1:
		i = 3
	elif subplot == 5:
		i = 4
	elif subplot == 9:
		i = 5

	ax = fig.add_subplot(3,3,subplot)

	# plt.hist([real_data[:,indexes[j]],gen_raw[:,indexes[j]]],bins=75,range=[min_values_posx[j],max_values_posx[j]],histtype='step',linewidth=3,color=['#094F9A','#BA1219'])
	plt.hist([real_data[:,i],gen_boost[:,i]],bins=75,range=[min_values_posx[i],max_values_posx[i]],histtype='step',linewidth=3,color=['#4772FF','#FF5959'])

	plt.yscale('log')

	ax.text(0.95,0.9, 'Log-scale',horizontalalignment='right',verticalalignment='center',transform=ax.transAxes,size=18)
	plt.xticks([])
	plt.yticks([])
	# ax.set_xticks([])
	# ax.set_yticks([])
	ax.tick_params(direction='in') 



plt.subplots_adjust(bottom=0.1, right=0.85, left=0.075, top=0.9, wspace = 0,hspace = 0)

plt.savefig('THESIS_PLOTS/BOOSTING_EXAMPLE_2.pdf',bbox_inches='tight')
plt.close('all')


quit()







































cax = plt.axes([0.9, 0.4333333333+0.1333333333, 0.01, 0.3])
cbar = plt.colorbar(cax=cax)
cbar.ax.set_ylabel('GAN Generated', rotation=270,fontsize=30,labelpad=40)
cbar.ax.tick_params(labelsize=30)

subplot = 1
# indexes = [0,1,2,3,4,5]
for j in range(3, 6):
	# print(i)
	subplot = j + 1 - 3
	subplot += 6 * (j - 3 + 1)
	add_6 = 0
	for i in range(j+1, 6):
		if add_6 != 0:
			subplot += 6 - 3
		print(i,j,subplot)

		# plt.subplot(6,6,subplot)
		ax = fig.add_subplot(3,3,subplot)
		plt.hist2d(real_data[:,indexes[j]],real_data[:,indexes[i]],bins=75,norm=LogNorm(),cmap=plt.cm.Blues_r,range=[[min_values_posx[j],max_values_posx[j]],[min_values_posx[i],max_values_posx[i]]],vmin=1,vmax=12107847)
		# plt.hist2d(real_data[:,indexes[j]],real_data[:,indexes[i]],bins=75,norm=LogNorm(),cmap=plt.cm.Blues_r,range=[[min_values_posx[j],max_values_posx[j]],[min_values_posx[i],max_values_posx[i]]],vmin=1,vmax=14471275.0)
		hist = np.histogram2d(gen_raw[:,indexes[j]],gen_raw[:,indexes[i]],bins=75,range=[[min_values_posx[j],max_values_posx[j]],[min_values_posx[i],max_values_posx[i]]])
		# plt.grid(axis='both', which='major',color='k',linestyle='--',alpha=0.3)
		if np.amax(hist[0]) > max_bin_value:
			max_bin_value = np.amax(hist[0])
			print(max_bin_value)

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

cax = plt.axes([0.9, 0.1333333333, 0.01, 0.3])
cbar = plt.colorbar(cax=cax)
cbar.ax.set_ylabel('Full Simulation', rotation=270,fontsize=30,labelpad=40)
# cbar.ax.set_xlabel('a.u.')
cbar.ax.tick_params(labelsize=30)

subplot = 1 - 7
for j in range(3, 6):
	# print(i)
	subplot += 7
	print(j,subplot)

	ax = fig.add_subplot(3,3,subplot)

	# plt.hist([real_data[:,indexes[j]],gen_raw[:,indexes[j]]],bins=75,range=[min_values_posx[j],max_values_posx[j]],histtype='step',linewidth=3,color=['#094F9A','#BA1219'])
	plt.hist([real_data[:,indexes[j]],gen_raw[:,indexes[j]]],bins=75,range=[min_values_posx[j],max_values_posx[j]],histtype='step',linewidth=3,color=['#4772FF','#FF5959'])

	if j in [0,1,2,3,4,5]:
		plt.yscale('log')

	ax.text(0.95,0.9, 'Log-scale',horizontalalignment='right',verticalalignment='center',transform=ax.transAxes,size=18)
	plt.xticks([])
	plt.yticks([])
	# ax.set_xticks([])
	# ax.set_yticks([])
	ax.tick_params(direction='in') 

print(max_bin_value)
plt.subplots_adjust(bottom=0.1, right=0.85, left=0.075, top=0.9, wspace = 0,hspace = 0)
plt.savefig('Triangle_ppt.png',bbox_inches='tight')
plt.close('all')

quit()



quit()





print(np.shape(real_data), np.shape(gen_raw))

real_data = np.take(real_data,np.random.permutation(real_data.shape[0]),axis=0,out=real_data)

gen_raw = np.take(gen_raw,np.random.permutation(gen_raw.shape[0]),axis=0,out=gen_raw)


real_data = real_data[:np.shape(gen_raw)[0]]



max_gan = [np.max(gen_raw[:,0]),np.max(gen_raw[:,1]),np.max(gen_raw[:,2]),np.max(gen_raw[:,3]),np.max(gen_raw[:,4]),np.max(gen_raw[:,5])]
min_gan = [np.min(gen_raw[:,0]),np.min(gen_raw[:,1]),np.min(gen_raw[:,2]),np.min(gen_raw[:,3]),np.min(gen_raw[:,4]),np.min(gen_raw[:,5])]

max_thomas = [np.max(real_data[:,0]),np.max(real_data[:,1]),np.max(real_data[:,2]),np.max(real_data[:,3]),np.max(real_data[:,4]),np.max(real_data[:,5])]
min_thomas = [np.min(real_data[:,0]),np.min(real_data[:,1]),np.min(real_data[:,2]),np.min(real_data[:,3]),np.min(real_data[:,4]),np.min(real_data[:,5])]

max_values_posx = np.empty(6)
min_values_posx = np.empty(6)


for i in range(0, 6):
	max_values_posx[i] = np.max([max_gan[i],max_thomas[i]])
	min_values_posx[i] = np.min([min_gan[i],min_thomas[i]])


symmetric = [0,1,3,4]

for i in symmetric:
	values = [np.absolute(min_values_posx[i]),np.absolute(max_values_posx[i])]
	max_value = np.max(values)
	min_values_posx[i] = -1 * max_value
	max_values_posx[i] = 1 * max_value

# make x and y the same

min_values_posx[0] = min_values_posx[1]
max_values_posx[0] = max_values_posx[1]

min_values_posx[3] = min_values_posx[4]
max_values_posx[3] = max_values_posx[4]

min_values_posx[5] = 0







subplot = 0
indexes = [0,1,2,3,4,5]
labels_indexes=['x','y','z','$P_x$','$P_y$','$P_z$']
axislabels_indexes=['x (cm)','y (cm)','z (cm)','$P_x$ (GeV/c)','$P_y$ (GeV/c)','$P_z$ (GeV/c)']

# ticks_i=[[-5,0,5],[-5,0,5],[-7000,-6800,-6600],[-5,0,5],[-5,0,5],[100,200,300]]


max_bin_value = 1


fig = plt.figure(figsize=(24,24))
for i in range(0, 6):
	# print(i)
	subplot += i
	for j in range(i+1, 6):
		subplot += 1
		print(i,j,subplot+1+i)

		# plt.subplot(6,6,subplot+1+i)
		ax = fig.add_subplot(6,6,subplot+1+i)
		plt.hist2d(gen_raw[:,indexes[j]],gen_raw[:,indexes[i]],bins=75,norm=LogNorm(),cmap=plt.cm.Reds_r,range=[[min_values_posx[j],max_values_posx[j]],[min_values_posx[i],max_values_posx[i]]],vmin=1,vmax=12107847)
		# plt.hist2d(gen_raw[:,indexes[j]],gen_raw[:,indexes[i]],bins=75,norm=LogNorm(),cmap=plt.cm.Reds_r,range=[[min_values_posx[j],max_values_posx[j]],[min_values_posx[i],max_values_posx[i]]],vmin=1,vmax=14471275.0)
		hist = np.histogram2d(gen_raw[:,indexes[j]],gen_raw[:,indexes[i]],bins=75,range=[[min_values_posx[j],max_values_posx[j]],[min_values_posx[i],max_values_posx[i]]])
		# plt.grid(axis='both', which='major',color='k',linestyle='--',alpha=0.3)
		if np.amax(hist[0]) > max_bin_value:
			max_bin_value = np.amax(hist[0])
			print(max_bin_value)

		# plt.xticks(ticks_i[j],ticks_i[j])
		# plt.yticks(ticks_i[i],ticks_i[i])

		# plt.tick_params(axis='y', which='both', labelsize=15)
		# plt.tick_params(axis='x', which='both', labelsize=15)

		if subplot+1+i in [2,3,4,5]:
			plt.yticks([])
			plt.gca().xaxis.tick_top()

			ax.set_xlabel(axislabels_indexes[j],fontsize=30)    
			ax.xaxis.set_label_position('top') 
		elif subplot+1+i == 6:
			plt.gca().xaxis.tick_top()
			plt.gca().yaxis.tick_right()
			ax.set_xlabel(axislabels_indexes[j],fontsize=30)    
			ax.xaxis.set_label_position('top')
			ax.set_ylabel(axislabels_indexes[i],fontsize=30)    
			ax.yaxis.set_label_position('right')
			continue
		elif subplot+1+i in [12,18,24,30,36]:
			plt.gca().yaxis.tick_right()
			plt.xticks([])
			ax.set_ylabel(axislabels_indexes[i],fontsize=30)    
			ax.yaxis.set_label_position('right')
		else:
			plt.xticks([])
			plt.yticks([])

		# ax.text(0.9,0.1, '%s'%labels_indexes[j],horizontalalignment='center',verticalalignment='center',transform=ax.transAxes,size=30)
		# ax.text(0.1,0.9, '%s'%labels_indexes[i],horizontalalignment='center',verticalalignment='center',transform=ax.transAxes,size=30)


cax = plt.axes([0.9, 0.4333333333+0.1333333333, 0.01, 0.3])
cbar = plt.colorbar(cax=cax)
cbar.ax.set_ylabel('GAN Generated', rotation=270,fontsize=30,labelpad=40)
cbar.ax.tick_params(labelsize=30)

subplot = 1
# indexes = [0,1,2,3,4,5]
for j in range(0, 6):
	# print(i)
	subplot = j + 1
	subplot += 6 * (j + 1)
	add_6 = 0
	for i in range(j+1, 6):
		if add_6 != 0:
			subplot += 6
		print(i,j,subplot)

		# plt.subplot(6,6,subplot)
		ax = fig.add_subplot(6,6,subplot)
		plt.hist2d(real_data[:,indexes[j]],real_data[:,indexes[i]],bins=75,norm=LogNorm(),cmap=plt.cm.Blues_r,range=[[min_values_posx[j],max_values_posx[j]],[min_values_posx[i],max_values_posx[i]]],vmin=1,vmax=12107847)
		# plt.hist2d(real_data[:,indexes[j]],real_data[:,indexes[i]],bins=75,norm=LogNorm(),cmap=plt.cm.Blues_r,range=[[min_values_posx[j],max_values_posx[j]],[min_values_posx[i],max_values_posx[i]]],vmin=1,vmax=14471275.0)
		hist = np.histogram2d(gen_raw[:,indexes[j]],gen_raw[:,indexes[i]],bins=75,range=[[min_values_posx[j],max_values_posx[j]],[min_values_posx[i],max_values_posx[i]]])
		# plt.grid(axis='both', which='major',color='k',linestyle='--',alpha=0.3)
		if np.amax(hist[0]) > max_bin_value:
			max_bin_value = np.amax(hist[0])
			print(max_bin_value)

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

cax = plt.axes([0.9, 0.1333333333, 0.01, 0.3])
cbar = plt.colorbar(cax=cax)
cbar.ax.set_ylabel('Full Simulation', rotation=270,fontsize=30,labelpad=40)
# cbar.ax.set_xlabel('a.u.')
cbar.ax.tick_params(labelsize=30)

subplot = 1 - 7
for j in range(0, 6):
	# print(i)
	subplot += 7
	print(j,subplot)

	ax = fig.add_subplot(6,6,subplot)

	# plt.hist([real_data[:,indexes[j]],gen_raw[:,indexes[j]]],bins=75,range=[min_values_posx[j],max_values_posx[j]],histtype='step',linewidth=3,color=['#094F9A','#BA1219'])
	plt.hist([real_data[:,indexes[j]],gen_raw[:,indexes[j]]],bins=75,range=[min_values_posx[j],max_values_posx[j]],histtype='step',linewidth=3,color=['#4772FF','#FF5959'])

	if j in [0,1,2,3,4,5]:
		plt.yscale('log')

	ax.text(0.95,0.9, 'Log-scale',horizontalalignment='right',verticalalignment='center',transform=ax.transAxes,size=18)
	plt.xticks([])
	plt.yticks([])
	# ax.set_xticks([])
	# ax.set_yticks([])
	ax.tick_params(direction='in') 

print(max_bin_value)
plt.subplots_adjust(bottom=0.1, right=0.85, left=0.075, top=0.9, wspace = 0,hspace = 0)
plt.savefig('Triangle_raw.png',bbox_inches='tight')
plt.close('all')




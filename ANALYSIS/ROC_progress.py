import numpy as np

import matplotlib.pyplot as plt

import glob

plt.rc('text', usetex=True)
plt.rcParams['savefig.dpi'] = 250
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rcParams.update({'font.size': 15})


def get_plotting_data(file_name):

	files = glob.glob('/Users/am13743/Aux_GAN_thesis/THESIS_ITERATION/BLUE_CRYSTAL_RESULTS/ROC_testing/%s/FoM_ROC_AUC_SCORE_list*'%file_name)

	for file_index, file in enumerate(files):

		if file_index == 0:
			current = np.load(file)
			results = np.empty((0,np.shape(current)[0],3))
		try:
			current = np.expand_dims(np.load(file),0)
			results = np.concatenate((results,current),axis=0)
		except:
			print('Incomplete file...')

	results_raw = results.copy()

	for result in results:
		
		value = 1
		for i in range(0, np.shape(result)[0]):
			if result[i][1] < value: value = result[i][1]
			result[i][1] = value

	mean = np.mean(results,axis=0)
	error = np.std(results,axis=0)/np.shape(results)[0]

	x = mean[:,0]
	y = mean[:,1]
	y_err = error[:,1]

	upper = y + y_err
	lower = y - y_err

	return x, y, y_err, upper, lower, results, results_raw


colours = ['tab:blue','tab:orange','tab:green','tab:red']

for plot_index, plot in enumerate(['GAN_1D','GAN_4D','Vanilla','Vanilla_OLD']):

	x, y, y_err, upper, lower, results, results_raw = get_plotting_data(plot)

	if plot == 'GAN_1D':
		label = 'Aux GAN 1D'
	if plot == 'GAN_4D':
		label = 'Aux GAN 4D'
	if plot == 'Vanilla':
		label = r'Vanilla $P_t$, $\theta_p$'
	if plot == 'Vanilla_OLD':
		label = r'Vanilla $P_x$, $P_y$'

	plt.plot(x,y,label=label,c=colours[plot_index],linewidth=2)
	for run in results_raw:
		plt.plot(run[:,0], run[:,1],alpha=0.15,c=colours[plot_index])

plt.ylim(0.5,1.)
plt.xlim(0,200000)
plt.legend(frameon=False)
plt.ylabel('ROC AUC FoM')
plt.xlabel('Training Steps')
plt.savefig('THESIS_PLOTS/ROC_progress.pdf',bbox_inches='tight')
# plt.show()






# -*- coding: utf-8 -*-
#### Imports
if True:
	print ("Starting")

	import matplotlib as mpl
	import os, time, sys
	import_seaborn = True

	
	workdir = os.path.dirname(os.path.realpath(__file__))
	data_dir=workdir+"/data"
	
	os.chdir(workdir)

	# can be removed
	OS = "Mac"

	import numpy as np
	import numpy.random as rnd
	import matplotlib.pyplot as plt
	import tensorflow as tf
	from pandas import DataFrame, Series,read_csv
	from math import exp, sqrt, sin, pi, cos
	

	from Logger import *
	from RBM_Functions import *


	np.set_printoptions(precision=3)


	### import seaborn? ###
	if import_seaborn:
		import seaborn

		seaborn.set(font_scale=1.2)
		seaborn.set_style("ticks",
			{
			'axes.grid':            True,
			'grid.linestyle':       u':',
			'legend.numpoints':     1,
			'legend.scatterpoints': 1,
			'axes.linewidth':       1.0,
			'xtick.direction':      'in',
			'ytick.direction':      'in',
			'xtick.major.size': 	5,
			'xtick.minor.size': 	2,
			'legend.frameon':       True,
			'ytick.major.size': 	5,
			'ytick.minor.size': 	2
			})

	mpl.rcParams["image.cmap"] = "gray"
	mpl.rcParams["grid.linewidth"] = 0.5
	mpl.rcParams["lines.linewidth"] = 1.25
	mpl.rcParams["font.family"]= "serif"

	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
		# plt.rcParams['image.cmap'] = 'coolwarm'
		# seaborn.set_palette(seaborn.color_palette("Set2", 10))

	log = Logger(True)


	time_now = time.asctime()
	time_now = time_now.replace(":", "-")
	time_now = time_now.replace(" ", "_")





#### Get the arguments list from terminal
additional_args = sys.argv[1:]

###########################################################################################################
#### Load User Settings ###
if len(additional_args)>0:
	log.out("Loading Settings from ", additional_args[0])
	Settings = __import__(additional_args[0])
else:
	import DefaultSettings as Settings

reload(Settings)

# rename 
UserSettings = Settings.UserSettings

# path were results are saved 
saveto_path = data_dir+"/"+time_now+"_"+str(UserSettings["DBM_SHAPE"])
if len(additional_args) > 0:
	saveto_path  += " - " + str(additional_args)

## open the logger-file
if UserSettings["DO_SAVE_TO_FILE"]:
	os.makedirs(saveto_path)
	log.open(saveto_path)


### modify the parameters with additional_args
if len(additional_args) > 0:
	try:
		first_param = int(additional_args[1])
		if len(additional_args) == 3:
			second_param = int(additional_args[2])
		log.out("Additional Arguments:", first_param, second_param)
		UserSettings["TEMP_START"] = second_param
		UserSettings["TEMP_MIN"] = second_param
		log.out(UserSettings)
	except:
		log.out("ERROR: Not using additional args!")



# load UserSettings into globals
log.out("For now, copying UserSettings into globals()")
for key in UserSettings:
	globals()[key] = UserSettings[key]

# better name for subset:
subset = SUBSPACE





###########################################################################################################
## set the seed for numpy.random
rnd.seed(UserSettings["SEED"])




## Error checking
if DO_LOAD_FROM_FILE and np.any(np.fromstring(PATHSUFFIX[26:].split("]")[0],sep=",",dtype=int) != DBM_SHAPE):
	log.out("Error: DBM Shape != Loaded Shape!")
	raise ValueError("DBM Shape != Loaded Shape!")


######### DBM #############################################################################################
DBM = DBM_class(shape = DBM_SHAPE,
				liveplot = 0,
				classification = DO_CLASSIFICATION,
				UserSettings = UserSettings,
				log = log,
				)

###########################################################################################################
#### Sessions ####
log.reset()
log.info(time_now)


DBM.pretrain(train_data)


if DO_TRAINING:
	log.start("DBM Train Session")


	with tf.Session() as sess:

		for run in range(N_EPOCHS_TRAIN):

			log.start("Run %i"%run)


			### start a train epoch
			DBM.train(	train_data  = train_data,
						train_label = train_label if LOAD_MNIST else None,
						train_label_attention = train_label_attention_side if LOAD_MNIST else None,
						num_batches = N_BATCHES_TRAIN,
						cont        = run)

			### test session while training
			if run!=N_EPOCHS_TRAIN-1 and run%TEST_EVERY_EPOCH==0:
				DBM.test(test_data, 
						test_label if LOAD_MNIST else None, 
						N               = 30,  # sample ist aus random werten, also mindestens 2 sample machen
						create_conf_mat = 0,
						temp_start      = DBM.temp,
						temp_end        = DBM.temp
						)

			# 	if LOAD_MNIST:
			# 		log.out("Creating Backup of Parameters")
			# 		DBM.backup_params()




			log.end()

	DBM.train_time=log.end()
	log.reset()

# last test session
if DO_TESTING:
	with tf.Session() as sess:
		DBM.test(test_data,
					test_label if LOAD_MNIST else None, 
					N               = 100,  # sample ist aus random werten, also mindestens 2 sample machen
					create_conf_mat = 0,
					temp_start      = DBM.temp,
					temp_end        = DBM.temp)
	# save_firerates_to_file(DBM.firerate_test,saveto_path+"/FireratesTest")


if DO_GEN_IMAGES:
	with tf.Session() as sess:
		log.start("Generation Session")

		if DO_LOAD_FROM_FILE and not DO_TRAINING:
			DBM.load_from_file(workdir+"/data/"+PATHSUFFIX,override_params=1)

		nn                = 10		# grid with nn^2 plots (has to stay 10 for now)
		DBM.batchsize     = nn**2

		DBM.graph_init("gibbs")
		DBM.import_()

		label_clamp = np.zeros([DBM.batchsize,10])
		# set the correct label value
		for j in range(10):
			label_clamp[10*j:10*j+10,j] = 1


		generated_img = DBM.gibbs_sampling(label_clamp,
							1000,
							DBM.temp, DBM.temp,
							999, 999,
							mode     = UserSettings["FREERUN_MODE"],
							subspace = [],
							liveplot = 0,
							l_input = None)

		# neurons with good weights
		if 0:
			layer = 3
			n_plots = 7
			# neurons with good firerate
			large_weight, classes = np.where(DBM.w_np[-1]>0.02)
			good_firerates = np.where((np.mean(DBM.layer_save_generate[layer][-1][:],0)>0.08)&(np.mean(DBM.layer_save_generate[layer][-1][:],0)<0.2))[0]
			neuron_ind_ = []
			classes_filtered = []
			for i in range(len(large_weight)):
				if large_weight[i] in good_firerates and large_weight[i] not in neuron_ind_:
					neuron_ind_.append(large_weight[i])
					classes_filtered.append(classes[i])

			fig, ax = plt.subplots(1,n_plots,figsize = (12,2.5))
			fig.subplots_adjust(wspace = 0.3, left = 0.1, right=0.97)
			# input for neuron on all images over time (shape : [time,batchsize])
			for i,neuron_ind in enumerate(neuron_ind_[:n_plots]):
				log.out(neuron_ind)
				log.out(classes[i])
				bu_input, td_input = follow_neuron_input(layer, neuron_ind, DBM.layer_save_generate, DBM)
				k = range(classes_filtered[i]*10,(classes_filtered[i]+1)*10);
				# ax[i].plot();
				seaborn.tsplot(bu_input[:,k].T/td_input[:,k].T, ci=[1], ax = ax[i]);
				seaborn.tsplot(bu_input[:,k].T/td_input[:,k].T, err_style = "unit_traces", ax = ax[i]);
				ax[i].set_xlabel("Time step "+r"$t$")
				ax[i].set_title("Neuron "+str(neuron_ind))
			ax[0].set_ylabel(r"$h_{b}/h_{t}$")

			fig,ax = plt.subplots(1,1)
			bu_input, td_input = follow_neuron_input(layer, neuron_ind_[0], DBM.layer_save_generate, DBM)
			bu_input = []
			td_input = []
			ks = []
			for i,neuron_ind in enumerate(neuron_ind_[:]):
				bu_input_, td_input_ = follow_neuron_input(layer, neuron_ind, DBM.layer_save_generate, DBM)
				k = range(classes_filtered[0]*10,(classes_filtered[0]+1)*10)
				bu_input.append(np.mean(bu_input_[:,range(classes_filtered[i]*10,(classes_filtered[i]+1)*10)],1))
				td_input.append(np.mean(td_input_[:,range(classes_filtered[i]*10,(classes_filtered[i]+1)*10)],1))
			bu_input = np.array(bu_input)
			td_input = np.array(td_input)
			bu_input *= 1./len(neuron_ind_)
			td_input *= 1./len(neuron_ind_)
			seaborn.tsplot(bu_input/td_input, err_style = "unit_traces", ci=[100], ax = ax);
			ax.set_ylabel(r"$|h_{b}/h_{t}|$")
			ax.set_xlabel("Time step "+r"$t$")

		#############################################################
		## plot the images (visible layers)
		fig,ax = plt.subplots(nn,nn)
		m = 0
		for i in range(nn):
			for j in range(nn):
				ax[i,j].matshow(generated_img[m].reshape(int(sqrt(DBM_SHAPE[0])), int(sqrt(DBM_SHAPE[0]))))
				ax[i,j].set_xticks([])
				ax[i,j].set_yticks([])
				ax[i,j].grid(False)
				m += 1
		save_fig(saveto_path+"/generated_img.png",DO_SAVE_TO_FILE)

		## plot the enrgies
		fig_en,ax_en = plt.subplots(nn,nn,figsize=(13,10),sharex="col",sharey="row")
		m=0
		for i in range(nn):
			for j in range(nn):
				ax_en[i,j].plot(DBM.energy_generate[:,m])
				m += 1
				ax_en[-1,j].set_xlabel("Timestep t")
			ax_en[i,0].set_ylabel("Energy")
		plt.tight_layout()
		save_fig(saveto_path+"/generated_energy.png",DO_SAVE_TO_FILE)
		log.end()


if DO_CONTEXT:

	if SEED != None:
		log.start("Context Session with SEED: %i"%DBM.SEED)
	else:
		log.start("Context Session with no SEED")

	if DO_LOAD_FROM_FILE and not DO_TRAINING:
		DBM.load_from_file(workdir+"/data/"+PATHSUFFIX,override_params=1)

	
	log.info("Subspace: ", subset)


	# loop through images from all wrong classsified images and find al images that are in subset
	index_for_number_gibbs=[]
	for i in range(10000):

		## find the digit that was presented
		digit = np.where(test_label[i])[0][0]

		## set desired digit range
		if digit in subset:
			index_for_number_gibbs.append(i)

	log.info("Found %i Images"%len(index_for_number_gibbs))

	
	# create graph
	DBM.batchsize=len(index_for_number_gibbs)
	if DBM.batchsize==0:
		raise ValueError("No Images found")


	log.start("Sampling data")
	with tf.Session() as sess:
		# first session with no context applied (see param "subset" = "all")
		DBM.graph_init("gibbs")
		DBM.import_()


		# calculte h2 firerates over all gibbs_steps
		h2_no_context = DBM.gibbs_sampling(test_data[index_for_number_gibbs[:]], 100,
							DBM.temp , DBM.temp,
							999, 999,
							mode     = "context",
							subspace = "all",
							liveplot = 0)

	with tf.Session() as sess:
		# second session (random vars are the same as 1. session) with context applied (see param "subset")
		DBM.graph_init("gibbs")
		DBM.import_()
		rnd.seed(UserSettings["SEED"])

		# # with context
		h2_context = DBM.gibbs_sampling(test_data[index_for_number_gibbs[:]], 100,
							DBM.temp , DBM.temp,
							999, 999,
							mode     = "context",
							subspace = subset,
							liveplot = 0)
	log.end()


	# append h2 activity to array, but only the unit that corresponst to the given digit picture
	desired_digits_c  = []
	desired_digits_nc = []
	wrong_digits_c    = []
	wrong_digits_nc   = []

	correct_maxis_c    = []
	correct_maxis_nc   = []
	incorrect_maxis_c  = []
	incorrect_maxis_nc = []

	wrongs_outside_subspace_c = 0
	wrongs_outside_subspace_nc = 0

	hist_data    = np.zeros([10,1,10]).tolist()
	hist_data_nc = np.zeros([10,1,10]).tolist()

	for i,d in enumerate(index_for_number_gibbs):
		digit = np.where( test_label[d] == 1 )[0][0]

		hist_data[digit].append( h2_context[i].tolist() )
		hist_data_nc[digit].append( h2_no_context[i].tolist() )

		### count how many got right (with context)
		## but only count the labels within subspace
		maxi_c         = h2_context[i][subset[:]].max()
		maxi_all_pos_c = np.where(h2_context[i] == h2_context[i].max())[0][0]
		max_pos_c      = np.where(h2_context[i] == maxi_c)[0][0]
		if max_pos_c   == digit:
			correct_maxis_c.append(maxi_c)
		else:
			if maxi_all_pos_c  not  in  subset:
				wrongs_outside_subspace_c += 1
			incorrect_maxis_c.append(maxi_c)

		### count how many got right (no context)
		## but only count the labels within subset
		maxi_nc     = h2_no_context[i][subset[:]].max()
		maxi_all_pos_nc = np.where(h2_no_context[i]==h2_no_context[i].max())[0][0]
		max_pos_nc  = np.where(h2_no_context[i] == maxi_nc)[0][0]
		if max_pos_nc == digit:
			correct_maxis_nc.append(maxi_nc)
		else:
			if maxi_all_pos_nc  not in  subset:
				wrongs_outside_subspace_nc += 1
			incorrect_maxis_nc.append(maxi_nc)

		desired_digits_c.append(h2_context[i,digit])
		desired_digits_nc.append(h2_no_context[i,digit])

		wrong_digits_c.append(np.mean(h2_context[i,digit+1:])+np.mean(h2_context[i,:digit]))
		wrong_digits_nc.append(np.mean(h2_no_context[i,digit+1:])+np.mean(h2_context[i,:digit]))



	log.info("Inorrect Context:" , len(incorrect_maxis_c),"/",round(100*len(incorrect_maxis_c)/float(len(index_for_number_gibbs)),2),"%")
	log.info("Inorrect No Context:" , len(incorrect_maxis_nc),"/",round(100*len(incorrect_maxis_nc)/float(len(index_for_number_gibbs)),2),"%")
	log.info("Diff:     ", len(incorrect_maxis_nc)-len(incorrect_maxis_c))
	log.info("Outside subset (c/nc):",wrongs_outside_subspace_c,",", wrongs_outside_subspace_nc)
	log.out("Means: Correct // Wrong (c/nc): \n \t \t ", 	round(np.mean(correct_maxis_c),4),
															round(np.mean(correct_maxis_nc),4), "//",
															round(np.mean(incorrect_maxis_c),4),
															round(np.mean(incorrect_maxis_nc),4)
			)
	## save to file
	if DO_SAVE_TO_FILE:
		file_gs = open(saveto_path+"/context_results.txt","w")
		file_gs.write("Found images: %i"%len(index_for_number_gibbs)+"\n")
		file_gs.write("Inorrect Context: " + str(len(incorrect_maxis_c))+"/"+str(round(100*len(incorrect_maxis_c)/float(len(index_for_number_gibbs)),2))+"%"+"\n")
		file_gs.write("Inorrect No Context: " + str(len(incorrect_maxis_nc))+"/"+str(round(100*len(incorrect_maxis_nc)/float(len(index_for_number_gibbs)),2))+"%"+"\n")
		file_gs.write("Diff: "+str(len(incorrect_maxis_nc)-len(incorrect_maxis_c))+"\n")
		file_gs.write("Outside subspace (c/nc): "+str(wrongs_outside_subspace_c)+","+ str(wrongs_outside_subspace_nc))
		# save hist data
		os.makedirs(saveto_path+"/context_hist")
		for cl in range(10):
			np.savetxt(saveto_path+"/context_hist/hist_data_c_digit_%i"%cl,np.array(hist_data[cl]))
			np.savetxt(saveto_path+"/context_hist/hist_data_nc_digit_%i"%cl,np.array(hist_data_nc[cl]))
		file_gs.close()



	log.end() #end session


if DO_NOISE_STAB:
	with tf.Session() as sess:
		plt.figure()
		my_pal=["#FF3045","#77d846","#466dd8","#ffa700","#48e8ff","#a431e5","#333333","#a5a5a5","#ecbdf9","#b1f6b6"]
		noise_h2_,v_noise_recon,v_noise=DBM.test_noise_stability(test_data[0:10], test_label[0:10],20)
		# with seaborn.color_palette(my_pal, 10):
		# 	for i in range(10):
		# 		plt.plot(smooth(noise_h2_[:,0,i],10),label=str(i))
		# 	plt.legend()
		fig,ax=plt.subplots(2,10,figsize=(10,4))
		for i in range(10):
			ax[0,i].matshow(v_noise[i].reshape(28,28))
			ax[1,i].matshow(v_noise_recon[i].reshape(28,28))
			ax[0,i].set_yticks([])
			ax[1,i].set_yticks([])

		plt.tight_layout(pad=0.0)


if DO_SAVE_TO_FILE:
	DBM.write_to_file()



####################################################################################################################################
#### Plot
# Plot the Weights, Errors and other informations
h1_shape = int(sqrt(DBM.SHAPE[1]))

log.out("Plotting...")

if DO_TRAINING:
	# plot w1 as image
	fig = plt.figure(figsize=(9,9))
	if DBM.type() == "DBM":
		map1 = plt.matshow(tile(DBM.w_np[0]), cmap = "gray", fignum = fig.number)
	else:
		map1 = plt.matshow(tile_attention(DBM.w_np[0]), cmap = "gray", fignum = fig.number)
	plt.colorbar(map1)
	plt.grid(False)
	plt.title("W %i"%0)
	save_fig(saveto_path+"/weights_img.pdf", DO_SAVE_TO_FILE)

	# plot layer diversity
	try:
		plt.figure("Layer diversity train")
		for i in range(DBM.n_layers):
			label_str = get_layer_label(DBM.type(),DBM.n_layers, i)
			y = smooth(np.array(DBM.layer_diversity_train)[:,i],10)
			plt.plot(DBM.updates[:len(y)],y,label=label_str,alpha=0.7)
			plt.legend()
		plt.xlabel("Update Number")
		plt.ylabel("Deviation")
		save_fig(saveto_path+"/layer_diversity.png", DO_SAVE_TO_FILE)
	except:
		print("Could not plot layer diversity.")
	plt.figure("Errors")
	## train errors
	# plt.plot(DBM.updates,DBM.recon_error_train,".",label="Recon Error Train",alpha=0.2)
	if DBM.classification:
		plt.plot(DBM.updates,DBM.class_error_train,".",label="Class Error Train",alpha=0.2)
	## test errors
	x = np.array(DBM.save_dict["Test_Epoch"]) * N_BATCHES_TRAIN
	# plt.plot(x,DBM.save_dict["Recon_Error"],"o--",label="Recon Error Test")
	if DBM.classification:
		plt.plot(x,DBM.save_dict["Class_Error"],"^--",label="Class Error Test")
		try:
			plt.plot(x[:-1],DBM.save_dict["Class_Error_Train_Data"],"x--",label="Class Error Test\nTrain Data")
		except:
			pass
	plt.legend(loc="best")
	plt.xlabel("Update Number")
	plt.ylabel("Mean Square Error")
	save_fig(saveto_path+"/errors.png", DO_SAVE_TO_FILE)


	# plot all other weights as hists
	log.out("Plotting Weights histograms")
	n_weights = len(DBM.w_np)
	fig,ax    = plt.subplots(n_weights,1,figsize=(8,10),sharex="col")
	for i in range(n_weights):
		if n_weights>1:
			seaborn.distplot(DBM.w_np[i].flatten(),rug=False,bins=60,ax=ax[i],label="After Training")
			ylim = ax[i].get_ylim()
			ax[i].axvline(DBM.w_np[i].max(),0,0.2, linestyle="-", color="k")
			ax[i].axvline(DBM.w_np[i].min(),0,0.2, linestyle="-", color="k")
			# ax[i].hist((DBM.w_np[i]).flatten(),bins=60,alpha=0.5,label="After Training")
			ax[i].set_title("W %i"%i)
			# ax[i].set_ylim(-ylim[1]/5,ylim[1])
			ax[i].legend()
		else:
			ax.hist((DBM.w_np[i]).flatten(),bins=60,alpha=0.5,label="After Training")
			ax.set_title("W %i"%i)
			ax.legend()

		try:
			seaborn.distplot(DBM.w_np_old[i].flatten(),rug=False,bins=60,ax=ax[i],label="Before Training",color="r")
			# ax[i].hist((DBM.w_np_old[i]).flatten(),color="r",bins=60,alpha=0.5,label="Before Training")
		except:
			pass
	plt.tight_layout()
	save_fig(saveto_path+"/weights_hist.pdf", DO_SAVE_TO_FILE)
	try:
		# plot change in w1
		fig=plt.figure(figsize=(9,9))
		plt.matshow(tile(DBM.w_np[0]-DBM.w_np_old[0]),fignum=fig.number)
		plt.colorbar()
		plt.title("Change in W1")
		save_fig(saveto_path+"/weights_change.pdf", DO_SAVE_TO_FILE)
	except:
		plt.close(fig)

	# plot freerun diffs
	fig,ax = plt.subplots(1,1)
	for i in range(DBM.n_layers):
		l_str = get_layer_label(DBM.type(),DBM.n_layers,i)
		ax.semilogy(DBM.updates, DBM.freerun_diff_train[:,i], label = l_str)
	plt.ylabel("log("+r"$\Delta L$"+")")
	plt.legend(ncol = 2, loc = "best")
	plt.xlabel("Update")
	save_fig(saveto_path+"/freerunn-diffs.png", DO_SAVE_TO_FILE)


	# plot train data (temp, diffs, learnrate, ..)
	fig,ax = plt.subplots(4,1,sharex="col",figsize=(8,8))

	ax[0].plot(DBM.save_dict["Temperature"],label="Temperature")
	ax[0].legend(loc="center left",bbox_to_anchor = (1.0,0.5))

	ax[0].set_ylabel("Temperature")

	ax[1].plot(DBM.save_dict["Learnrate"],label="Learnrate")
	ax[1].legend(loc="center left",bbox_to_anchor = (1.0,0.5))
	ax[1].set_ylabel("Learnrate")

	ax[2].set_ylabel("Mean")
	for i in range(len(DBM.w)):
		ax[2].plot(DBM.save_dict["W_mean_%i"%i],label="W %i"%i)
	ax[2].legend(loc="center left",bbox_to_anchor = (1.0,0.5))

	ax[3].set_ylabel("Diff")
	if DBM.epochs>2:
		for i in range(len(DBM.SHAPE)-1):
			ax[3].semilogy(DBM.save_dict["W_diff_%i"%i][1:],label = "W %i"%i)
		for i in range(len(DBM.SHAPE)):
			ax[3].semilogy(DBM.save_dict["Bias_diff_%i"%i][1:],label = "Bias %i"%i)

	ax[3].legend(loc="center left",bbox_to_anchor = (1.0,0.5))

	ax[-1].set_xlabel("Epoch")

	plt.subplots_adjust(bottom=None, right=0.73, left=0.2, top=None,
	            wspace=None, hspace=None)

	save_fig(saveto_path+"/learnr-temp.pdf", DO_SAVE_TO_FILE)




	plt.figure("Layer_activiations_train_run")
	for i in range(DBM.n_layers):
		label_str = get_layer_label(DBM.type(),DBM.n_layers, i)
		plt.plot(DBM.updates,np.array(DBM.layer_act_train)[:,i],label = label_str)
	plt.legend()
	plt.xlabel("Update Number")
	plt.ylabel("Train Layer Activity in %")
	save_fig(saveto_path+"/layer_act_train.png", DO_SAVE_TO_FILE)

if DO_TESTING:
	# plot some samples from the testdata
	fig3,ax3 = plt.subplots(len(DBM.SHAPE)+1,13,figsize=(16,8),sharey="row")
	for i in range(13):
		# plot the input
		if DBM.type() != "DBM_attention":
			ax3[0][i].matshow(test_data[i:i+1].reshape(int(sqrt(DBM.SHAPE[0])),int(sqrt(DBM.SHAPE[0]))))
			ax3[1][i].matshow(DBM.firerate_test[0][i].reshape(int(sqrt(DBM.SHAPE[0])),int(sqrt(DBM.SHAPE[0]))))
		else:
			ax3[0][i].matshow(test_data_attention[i].reshape(28*2,28))
			ax3[0][i].set_title(str(test_label_attention_side[i]))
			ax3[1][i].matshow(DBM.firerate_test[0][i].reshape(28*2,28))	
	
		ax3[0][i].set_yticks([])
		ax3[0][i].set_xticks([])
	
		ax3[1][i].set_yticks([])
		ax3[1][i].set_xticks([])

		#plot hidden layer
		for layer in DBM.get_hidden_layer_ind():
			try:
				ax3[layer+1][i].matshow(DBM.firerate_test[layer][i].reshape(int(sqrt(DBM.SHAPE[layer])),int(sqrt(DBM.SHAPE[layer]))))
				ax3[layer+1][i].set_yticks([])
				ax3[layer+1][i].set_xticks([])
			except:
				pass
		# plot the last layer
		if DBM.classification:
			if DBM.type() == "DBM_attention" and np.where(DBM.label_l_save[i]==DBM.label_l_save[i].max())[0] == np.where(test_label_attention_class[i] == 1)[0]:
				color = "g"
			elif DBM.type() == "DBM" and np.where(DBM.label_l_save[i]==DBM.label_l_save[i].max())[0][0] == np.where(test_label[i] == 1)[0][0]:
				color = "g"
			else:
				color = "r"
			ax3[-DBM.n_label_layer][i].bar(range(DBM.SHAPE[-DBM.n_label_layer]//label_mult),DBM.label_l_save[i],color=color)
			ax3[-DBM.n_label_layer][i].set_xticks(range(DBM.SHAPE[-DBM.n_label_layer]//label_mult))
			ax3[-DBM.n_label_layer][i].set_ylim(0,1)

			if DBM.type() == "DBM_attention":
				ax3[-1][i].bar(range(DBM.SHAPE[-1]),DBM.firerate_test[-1][i])
		else:
			ax3[-1][i].matshow(DBM.label_l_save[i].reshape(int(sqrt(DBM.SHAPE[-1])),int(sqrt(DBM.SHAPE[-1]))))
			ax3[-1][i].set_xticks([])
			ax3[-1][i].set_yticks([])

		#plot the reconstructed layer h1
		# ax3[5][i].matshow(DBM.rec_h1[i:i+1].reshape(int(sqrt(DBM.SHAPE[1])),int(sqrt(DBM.SHAPE[1]))))
		# plt.matshow(random_recon.reshape(int(sqrt(DBM.SHAPE[0])),int(sqrt(DBM.SHAPE[0]))))
	plt.tight_layout(pad=0.0)
	save_fig(saveto_path+"/examples.pdf", DO_SAVE_TO_FILE)


if LOAD_MNIST and DO_TESTING:

	## plot layer activities % test run
	plt.figure("Layer_activiations_test_run")
	for i in range(DBM.n_layers):
		label_str = get_layer_label(DBM.type(), DBM.n_layers, i)
		plt.plot(DBM.layer_act_test[:,i],label=label_str)
	plt.legend()
	plt.xlabel("timestep")
	plt.ylabel("Test Layer Activity in %")
	save_fig(saveto_path+"/layer_act_test.pdf", DO_SAVE_TO_FILE)

	# plot the layer diversity after test run
	plt.figure("Layer stddeviation across test batch",figsize=(5,4))
	plt.bar(range(DBM.n_layers),DBM.layer_diversity_test, yerr = DBM.layer_variance_test)
	plt.ylabel(r"$<\sigma_i>_{layer}$")
	plt.xlabel("Layer")
	plt.xticks(range(DBM.n_layers),[get_layer_label(DBM.type(), DBM.n_layers, i ,short=True) for i in range((DBM.n_layers))])
	plt.tight_layout()
	save_fig(saveto_path+"/layer_std_test_batch.pdf", DO_SAVE_TO_FILE)

	# firerates test run mean hist
	fig,ax = plt.subplots(1,DBM.n_layers-1,figsize=(10,2.75),sharex="row")
	for l in range(1,DBM.n_layers):
		ax[l-1].hist(np.mean(DBM.firerate_test[l],0),lw=0.1,edgecolor="k")
		ax[l-1].set_title(get_layer_label(DBM.type(), DBM.n_layers,l))
		ax[l-1].set_ylabel(r"$N$")
		ax[l-1].set_xlabel("Firerate")
	plt.tight_layout()
	save_fig(saveto_path+"/firerates_testrun.pdf", DO_SAVE_TO_FILE)


	# plot l_input_test
	fig,ax = plt.subplots(1,1,figsize=(4,4))
	color_m = 1
	m = 0
	max_y = 0
	for i in range(DBM.n_layers):
		# color = next(ax._get_lines.prop_cycler)['color'];
		color=np.array([0.6,0.6,0.6])

		for direc in range(2):
			if direc == 0:
				versch = -0.125
				color_m = 1
				hatch="///"
				label = "bottom up"
			else:
				versch = +0.125
				color_m = 1.4
				hatch="xx"
				label = "top down"

			max_y_ = np.max(np.mean(np.abs(DBM.l_input_test[i][direc])))
			if max_y_>max_y:
				max_y = max_y_
			ax.bar(m+versch, np.mean(np.abs(DBM.l_input_test[i][direc])), width = 0.25,
				color = color*color_m, linewidth = 1.0, edgecolor = "k",
				yerr  = np.mean(np.abs(DBM.l_var_test[i][direc])),
				)
			if i == 0:
				ax.bar(1,0,label=label,color=color*color_m, linewidth = 1.0, edgecolor = "k")

			ax.set_ylabel("Average Input")

		m+=1
		ax.set_xlabel("Layer")
		ax.set_ylim(0,max_y*1.2)
		ax.set_xticks(range(DBM.n_layers))
	ax.legend(loc="best")
	fig.tight_layout()
	save_fig(saveto_path+"/total_layer_input.pdf", DO_SAVE_TO_FILE)


	# plot l_input_test as hist over all units
	fig,ax = plt.subplots(DBM.n_layers,1,figsize=(7,10))
	for i in range(DBM.n_layers):
		max_x = 0
		ax_index = -(i+1)
		for direc in range(2):
			color = next(ax[i]._get_lines.prop_cycler)['color'];
			# color="r"
			label = "bottom up" if direc == 0 else "top down"
			data = np.array(DBM.hist_input_test[i][direc]).flatten()

			try:

				max_x_ = data.max()
				if max_x_>max_x:
					max_x = max_x_

				y,x,_ = ax[ax_index].hist(data,
					bins      = 50,
					label     = label,
					color     = color,
					linewidth = 0.2,
					edgecolor = "k",
					alpha     = 0.8,
					# density   = True#np.zeros_like(data)+1/data.size
					)


			except:
				pass

		# ax[ax_index].set_ytick(ax[i].get_yticks())
		ax[ax_index].set_xlim(-max_x,max_x)
		ax[ax_index].set_ylabel(r"$N$")
		ax[ax_index].ticklabel_format(style = 'sci',scilimits=(-2,2))
		ax[ax_index].set_title(get_layer_label(DBM.type(), DBM.n_layers,i,short=True))
		ax[ax_index].legend()
	ax[-1].set_xlabel("Input Strength")
	plt.tight_layout()
	save_fig(saveto_path+"/layer_input_hist.pdf", DO_SAVE_TO_FILE)

	# plot timeseries of every neuron while testrun (clamped v)
	# layer_save_test has shape : [time][layer][image][neuron]
	# k = 0 #which example image to pick
	# if not os.path.isdir(saveto_path+"/timeseries_testrun"):
	# 	os.makedirs(saveto_path+"/timeseries_testrun")

	# for layer in range(1,DBM.n_layers):
	# 	plt.matshow(DBM.layer_save_test[layer][:,k])
	# 	plt.xlabel("Time "+r"$t$")
	# 	plt.ylabel("Unit "+r"$i$")
	# 	save_fig(saveto_path+"/timeseries_testrun/timeseries_1image_layer_%i.png"%(layer+1),DO_SAVE_TO_FILE)

	# 	# plt the average over all test images
	# 	plt.matshow(np.mean(DBM.layer_save_test[layer][:,:],1))
	# 	plt.xlabel("Time "+r"$t$")
	# 	plt.ylabel("Unit "+r"$i$")

	# 	save_fig(saveto_path+"/timeseries_testrun/timeseries_av_layer_%i.png"%(layer+1),DO_SAVE_TO_FILE)



	# plot only one digit
	if DBM.type() == "DBM":
		fig3,ax3 = plt.subplots(len(DBM.SHAPE)+1,10,figsize=(16,8),sharey="row")
		m=0
		for i in index_for_number_test_clear[8][0:10]:
			# plot the input
			ax3[0][m].matshow(test_data[i:i+1].reshape(int(sqrt(DBM.SHAPE[0])),int(sqrt(DBM.SHAPE[0]))))
			ax3[0][m].set_yticks([])
			ax3[0][m].set_xticks([])
			# plot the reconstructed image
			ax3[1][m].matshow(DBM.firerate_test[0][i].reshape(int(sqrt(DBM.SHAPE[0])),int(sqrt(DBM.SHAPE[0]))))
			ax3[1][m].set_yticks([])
			ax3[1][m].set_xticks([])

			#plot hidden layer
			for layer in DBM.get_hidden_layer_ind():
				try:
					ax3[layer+1][m].matshow(DBM.firerate_test[layer][i].reshape(int(sqrt(DBM.SHAPE[layer])),int(sqrt(DBM.SHAPE[layer]))))
					ax3[layer+1][m].set_yticks([])
					ax3[layer+1][m].set_xticks([])
				except:
					pass
			# plot the last layer
			if DBM.classification:
				ax3[-DBM.n_label_layer][m].bar(range(DBM.SHAPE[-DBM.n_label_layer]//label_mult),DBM.label_l_save[i])
				ax3[-DBM.n_label_layer][m].set_xticks(range(DBM.SHAPE[-DBM.n_label_layer]//label_mult))
				ax3[-DBM.n_label_layer][m].set_ylim(0,1)

				if DBM.type() == "DBM_attention":
					ax3[-1][m].bar(range(DBM.SHAPE[-1]),DBM.firerate_test[-1][i])
			else:
				ax3[-1][m].matshow(DBM.label_l_save[i].reshape(int(sqrt(DBM.SHAPE[-1])),int(sqrt(DBM.SHAPE[-1]))))
				ax3[-1][m].set_xticks([])
				ax3[-1][m].set_yticks([])
				#plot the reconstructed layer h1
				# ax4[5][m].matshow(DBM.rec_h1[i:i+1].reshape(int(sqrt(DBM.SHAPE[1])),int(sqrt(DBM.SHAPE[1]))))
				# plt.matshow(random_recon.reshape(int(sqrt(DBM.SHAPE[0])),int(sqrt(DBM.SHAPE[0]))))
			m+=1
		plt.tight_layout()
		save_fig(saveto_path+"/examples_one_digit.pdf", DO_SAVE_TO_FILE)


if DO_CONTEXT:

	# plot the variance of the layers for c/nc normed to nc and the firerates as hist
	log.out("Plotting variance diff c/nc")
	plt.figure()
	layer_str = [""]*DBM.n_layers
	for i in range(DBM.n_layers):
		diff = DBM.layer_diversity_c[i]/DBM.layer_diversity_nc[i]
		layer_str[i] = get_layer_label(DBM.type(), DBM.n_layers, i, short=True)
		plt.bar(i,diff)
	plt.xticks(range(DBM.n_layers),layer_str)
	plt.xlabel("Layer")
	plt.ylabel("Diversity")
	save_fig(saveto_path+"/context_l_diversity.pdf",DO_SAVE_TO_FILE)



	### plt histograms for each used digit
	fig,ax = plt.subplots(1,len(subset),figsize=(12,7),sharey="row")
	fig2,ax2 = plt.subplots(1,len(subset),figsize=(7,4),sharey="row")
	for i,digit in enumerate(subset):
		if len(hist_data_nc[digit])>1:
			y_nc = np.mean(np.array(hist_data_nc[digit][1:]),axis=0)
			y_c  = np.mean(np.array(hist_data[digit][1:]),axis=0)

			for j in range(10):
				if j in SUBSPACE:
					rel_change = (y_c[j]-y_nc[j])/(1-y_nc[j])
					c = "r" if j != digit else "g"
					ax2[i].bar(j,rel_change, color = c)

				if y_nc[j]>y_c[j]:
					ax[i].bar(j,y_nc[j],color=[0.8,0.1,0.1],linewidth=0.1,edgecolor="k")
					ax[i].bar(j,y_c[j] ,color=[0.1,0.7,0.1],linewidth=0.1,edgecolor="k")
				else:
					ax[i].bar(j,y_c[j] ,color=[0.1,0.7,0.1],linewidth=0.1,edgecolor="k")
					ax[i].bar(j,y_nc[j],color=[0.8,0.1,0.1],linewidth=0.1,edgecolor="k")

		ax[i].set_ylim([0,1])
		ax[i].set_title(str(digit))
		ax[i].set_xticks(range(10))
		ax2[i].set_xticks(range(len(subset)))
		ax2[i].set_title(str(digit))
		ax2[i].set_xlabel("Class")
	ax[-1].plot(0,0,color = [0.8,0.1,0.1], label="No Context")
	ax[-1].plot(0,0,color = [0.1,0.7,0.1], label="Context")
	ax[0].set_ylabel("Mean Predicted Label")
	ax2[0].set_ylabel("Relative Increase "+r"$\rho_{\mathrm{rel}}$")
	plt.legend(loc="center left",bbox_to_anchor = (1.0,0.5))
	fig.subplots_adjust(bottom=None, right=0.84, left=0.1, top=None,
	        wspace=None, hspace=None)
	fig2.tight_layout()

	save_fig(saveto_path+"/context_hists.pdf",DO_SAVE_TO_FILE)




	# plot time series
	fig,ax = plt.subplots(1,3,figsize=(13,4));
	for i in range(DBM.n_layers-1):
		color = next(ax[0]._get_lines.prop_cycler)['color'];
		# color="r"
		ax[0].plot(DBM.activity_nc[i],"--",color = color)
		ax[0].plot(DBM.activity_c[i],"-",color   = color)
		label_str = get_layer_label(DBM.type(),DBM.n_layers, i+1)
		ax[0].plot(0,0,color=color,label=label_str)
	ax[0].plot(1,1,"k--",label="No Context")
	ax[0].plot(1,1,"k-",label="with Context")
	# ax[0].legend(loc="upper right")
	ax[0].set_ylabel("Active Neurons in \%")
	ax[0].set_xlabel("Timestep")

	for i in range(DBM.n_layers-1):
		color = next(ax[1]._get_lines.prop_cycler)['color'];
		# color="r"

		ax[1].plot(range(1,len(DBM.layer_diff_gibbs_c[i])),DBM.layer_diff_gibbs_c[i, 1:],"-",color=color)
		ax[1].plot(range(1,len(DBM.layer_diff_gibbs_nc[i])),DBM.layer_diff_gibbs_nc[i, 1:],"--",color=color)
		label_str = get_layer_label(DBM.type(), DBM.n_layers, i+1)
		ax[1].plot(0,0,color=color,label=label_str)
	ax[1].plot(0,0,"k--",label="No Context")
	ax[1].plot(0,0,"k-",label="with Context")
	ax[1].legend(loc="center left",bbox_to_anchor = (2.5,0.5))
	ax[1].set_ylabel(r"$|$"+"Layer(t) - Layer(t-1)"+r"$|$")
	ax[1].set_xlabel("Timestep")

	ax[2].plot(DBM.class_error_gibbs_c,"-",color=color)
	ax[2].plot(DBM.class_error_gibbs_nc,"--",color=color)

	ax[2].plot(0,0,"k--",label="No Context")
	ax[2].plot(0,0,"k-",label="With Context")
	# ax[2].legend(loc="best")
	ax[2].set_ylabel("Class Error")
	ax[2].set_xlabel("Timestep")
	plt.subplots_adjust(bottom=0.19, right=0.83, left=0.1, top=None,
		            wspace=0.37, hspace=None)
	save_fig(saveto_path+"/context_time_series.pdf",DO_SAVE_TO_FILE)



	# plot layer input with and without context
	fig,ax  = plt.subplots(1,1)
	color_m = 1
	m       = 0
	max_y   = 0
	for i in range(DBM.n_layers):
		color = next(ax._get_lines.prop_cycler)['color'];
		# color="r"

		for direc in range(2):
			if direc == 0:
				versch = -0.125
				color_m = 1
			else:
				versch = +0.125
				color_m = 1.2

			data_c     = np.mean(np.abs(DBM.l_input_c[i][direc]))
			data_nc     = np.mean(np.abs(DBM.l_input_nc[i][direc]))
			filename = "/context_total_l_input_diff.pdf"


			ax.bar(m+versch, data_c-data_nc,width = 0.25, #hatch="///",
				 linewidth = 0.2, edgecolor = "k",

				)
			ax.set_ylabel("Aver Input")

		m+=1
		ax.set_xlabel("Layer")
	save_fig(saveto_path+filename, DO_SAVE_TO_FILE)




	# plot the unit variance and firerate c/nc
	fig_sig,ax_sig         = plt.subplots(2,DBM.n_layers-2,figsize=(10,5))
	fig_f,ax_f             = plt.subplots(2,DBM.n_layers-2,figsize=(10,5))
	biggest_var_change_ind = [None]*(DBM.n_layers)
	for l in DBM.get_hidden_layer_ind():
		layer_str = get_layer_label(DBM.type(), DBM.n_layers,l,short=True)

		delta_sigma = DBM.unit_diversity_c[l]-DBM.unit_diversity_nc[l]
		delta_f     = np.mean(DBM.firerate_c[l],0) - np.mean(DBM.firerate_nc[l],0)

		biggest_var_change_ind[l] = np.where(abs(delta_sigma) > sorted(abs(delta_sigma))[-11] )[0]
		rnd_index = rnd.randint(0,225,size=(11))


		big_var_change_hists_c  = calc_neuron_hist(rnd_index, DBM.firerate_c[l],   test_label[index_for_number_gibbs[:]], 0.5, len(subset))
		big_var_change_hists_nc = calc_neuron_hist(rnd_index, DBM.firerate_nc[l],  test_label[index_for_number_gibbs[:]], 0.5, len(subset))

		fig2, ax2 = plt.subplots(2,len(big_var_change_hists_c)//2,figsize=(8,4),sharey="row")
		m=0
		for j in range(2):
			for i in range(len(big_var_change_hists_c)//2):
				ax2[j,i].bar(np.array(subset)-0.17,big_var_change_hists_c[m], width = 0.35, color="g")
				ax2[j,i].bar(np.array(subset)+0.17,big_var_change_hists_nc[m], width = 0.35, color="r")
				ax2[j,i].set_title(r"$\Delta \sigma = $"+str(np.round(delta_sigma[rnd_index[m]],3)))
				ax2[-1,i].set_xlabel("Class")
				ax2[j,0].set_ylabel(r"$N$")
				ax2[j,i].set_xticks((subset))
				m+=1
		fig2.tight_layout()
		save_fig(saveto_path+"/big_var_change_l%i.pdf"%l, DO_SAVE_TO_FILE)
		if DO_SAVE_TO_FILE:
			plt.close(fig2)

		ax_f[0,l-1].hist(np.mean(DBM.firerate_nc[l],0),bins=30, lw = 0.2, edgecolor = "k")
		ax_f[1,l-1].hist(delta_f,bins=30, lw = 0.2, edgecolor = "k")

		ax_f[0,l-1].set_xlabel(r"$<f_{%s}^{\mathrm{nc}}>_\mathrm{dataset}$"%layer_str[1:-1])
		ax_f[1,l-1].set_xlabel(r"$\Delta f_{%s}$"%layer_str[1:-1])
		ax_f[0,l-1].set_ylabel("N",style= "italic")
		ax_f[1,l-1].set_ylabel("N",style= "italic")


		ax_sig[0,l-1].hist(DBM.unit_diversity_nc[l], bins=20, lw = 0.2, edgecolor = "k")
		ax_sig[1,l-1].hist(delta_sigma, bins=20, lw = 0.2, edgecolor = "k")

		ax_sig[0,l-1].set_xlabel(r"$<\sigma_{%s}^{\mathrm{nc}}>_\mathrm{dataset}$"%layer_str[1:-1])
		ax_sig[1,l-1].set_xlabel(r"$\Delta \sigma_{%s}$"%layer_str[1:-1])
		ax_sig[0,l-1].set_ylabel("N",style= "italic")
		ax_sig[1,l-1].set_ylabel("N",style= "italic")

		# ax_sig[0,l-1].hist(np.mean(DBM.firerate_nc[l-1][:],0), bins=20, alpha=0.7, label = "Without context", lw=0.2,edgecolor="k")
		# ax_sig[0,l-1].hist(np.mean(DBM.firerate_test[l][:],0),bins=20,alpha=0.7,label = "Testrun",lw=0.2,edgecolor="k")

		# plt.colorbar(ax_sig=ax_sig[l-1],mappable=mapp)#,cbarlabel="$\sigma_%s^c/\sigma_%s^{nc}$"%(layer_str,layer_str))
		# ax_sig[0,l-1].set_xlim([0,1])

	fig_sig.tight_layout()
	fig_f.tight_layout()
	save_fig(saveto_path+"/layer_f.pdf", DO_SAVE_TO_FILE)
	if DO_SAVE_TO_FILE:
		plt.close(fig_f)
	save_fig(saveto_path+"/layer_sig.pdf", DO_SAVE_TO_FILE)

	### look at neurons that where active outside subspace while testing and chekc if they got active during context
	# look which hists have their  max outside supspace
	# if DO_TESTING:
	# 	log.out("Searching neurons that fired outside subspace while testing")
	# 	max_neurons = 20
	# 	outside_subspace_ind = [[]*i for i in range(DBM.n_layers)]
	# 	for l in DBM.get_hidden_layer_ind():
	# 		# generate hisogramms for all neurons that fired reasonable often
	# 		hists_test = calc_neuron_hist(DBM.neuron_good_test_firerate_ind[l],DBM.firerate_test[l],test_label,0.5, 10)

	# 		# go through every hist and chekc if high values are outside subspace
	# 		for i in range(len(hists_test)):
	# 			where  =  np.where(hists_test[i]>hists_test[i].mean()+hists_test[i].std())[0]
	# 			for j in where:
	# 				if j not in subspace:
	# 					outside_subspace_ind[l].append(i)
	# 					break
	# 			if len(outside_subspace_ind[l]) >= max_neurons:
	# 				break


	# 		hists_c  = calc_neuron_hist(outside_subspace_ind[l], DBM.firerate_c[l-1],  test_label[index_for_number_gibbs[:]], 0.5, len(subspace))
	# 		hists_nc = calc_neuron_hist(outside_subspace_ind[l], DBM.firerate_nc[l-1], test_label[index_for_number_gibbs[:]], 0.5, len(subspace))
	# 		hists_t = calc_neuron_hist(outside_subspace_ind[l], DBM.firerate_test[l], test_label, 0.5, 10)
	# 		hists_c = np.array(hists_c)
	# 		hists_nc = np.array(hists_nc)
	# 		hists_t = np.array(hists_t)
	# 		log.out("Searching which of the found neurons also had a moderate firerate while gibbs sampling")
	# 		w_firerates = np.where((np.mean(DBM.firerate_c[l-1][:,outside_subspace_ind[l]],0)<0.4) & (np.mean(DBM.firerate_c[l-1][:,outside_subspace_ind[l]],0)>0.02))[0]

	# 		log.out("Plotting these neurons hists")
	# 		num_plots = int(sqrt(len(w_firerates)))
	# 		fig,ax = plt.subplots(num_plots,num_plots,sharex="col",sharey="row")
	# 		m=0
	# 		for i in range(num_plots):
	# 			for j in range(num_plots):
	# 				index = outside_subspace_ind[l][w_firerates[m]]
	# 				ax[i,j].bar(subspace,hists_c[w_firerates[m]],alpha=0.7,color=[0.0, 1, 0.5])
	# 				ax[i,j].bar(subspace,hists_nc[w_firerates[m]],alpha=0.7,color=[0.8, 0.3, 0.3],width=0.2)
	# 				ax[i,j].bar(range(10),hists_t[index],alpha=0.5,color=[1, 0.0, 0.0])
	# 				ax[i,j].set_xticks(range(10))
	# 				ax[-1,j].set_xlabel("Class")
	# 				ax[i,0].set_ylabel(r"$N$")
	# 				# ax[i,j].set_title(str(index)+" | "+ str(diffs[index]))
	# 				m+=1
	# 		fig.tight_layout()
	# 		save_fig(saveto_path+"/outisde_subspace_hists_l%i"%l, DO_SAVE_TO_FILE)

# if DO_GEN_IMAGES:
# 	# plot timeseries of every neuron while generate (clamped label)
# 	# layer_save_generate has shape : [time][layer][image][neuron]
# 	k = 0 #which example image to pick
# 	if not os.path.isdir(saveto_path+"/timeseries_generated"):
# 		os.makedirs(saveto_path+"/timeseries_generated")
# 	for layer in range(DBM.n_layers-1):
# 		# timeseries = []
# 		# timeseries_average = []
# 		# for i in range(len(DBM.layer_save_generate)):
# 		# 	timeseries.append(DBM.layer_save_generate[i][layer][k])
# 		# 	timeseries_average.append(np.mean(DBM.layer_save_generate[i][layer],0))


# 		# plot for image k
# 		plt.matshow(DBM.layer_save_generate[layer][:,k])
# 		plt.ylabel("Time "+r"$t$")
# 		plt.xlabel("Unit "+r"$i$")
# 		save_fig(saveto_path+"/timeseries_generated/timeseries_1image_layer_%i.png"%(layer+1),DO_SAVE_TO_FILE)

# 		# plt the average over all test images
# 		plt.matshow(np.mean(DBM.layer_save_generate[layer][:,:],1))
# 		plt.ylabel("Time "+r"$t$")
# 		plt.xlabel("Unit "+r"$i$")

# 		save_fig(saveto_path+"/timeseries_generated/timeseries_av_layer_%i.png"%(layer+1),DO_SAVE_TO_FILE)

## plot the receptive field of the ...
# get hidden layers of one sampled image as boolen
# h = []
# for l in range(1,len(DBM.SHAPE)-1):
# 	h.append(DBM.layer_save_generate[l][-1,0].astype(bool))
# v2 = DBM.layer_save_generate[-1][-1,0]
# h.append(sample_np(v2))
# # clean the weights
# w = []
# for l in range(len(DBM.w_np)):
# 	w.append(np.copy(DBM.w_np[l]))
# 	if l < len(DBM.w_np)-2:
# 		w[l][:,np.logical_not(h[l])] = 0
# # use the cleaned weights for calculation
# ww = np.dot(w[0],w[1])
# # ww = np.dot(ww,w[2])
# # ww = np.dot(ww,w[3])
# for i in range(2,len(DBM.SHAPE)-1):
# 	ww = np.dot(ww,w[i])

# fig,ax = plt.subplots(1,10)
# for i in range(10):
# 	ax[i].matshow(ww[:,i].reshape(28,28))
# 	ax[i].set_xticks([])
# 	ax[i].set_yticks([])
# 	ax[i].set_title(str(i))

log.out("Finished")
# log.close()
if DO_SHOW_PLOTS:
	plt.show()
else:
	plt.close()

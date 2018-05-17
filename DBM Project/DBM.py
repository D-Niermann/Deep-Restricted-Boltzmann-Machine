# -*- coding: utf-8 -*-
#### Imports
if True:
	# -*- coding: utf-8 -*-
	print ("Starting")
	
	import matplotlib as mpl
	import os,time,sys

	try: # if on macbook
		workdir="/Users/Niermann/Google Drive/Masterarbeit/Python/DBM Project"
		os.chdir(workdir)
		import_seaborn = True
	except: #else on university machine
		workdir="/home/dario/Dokumente/DBM Project"
		os.chdir(workdir)
		mpl.use("Agg") #use this to not display plots but save them
		import_seaborn = True

	data_dir=workdir+"/data"

	import numpy as np
	import numpy.random as rnd
	import matplotlib.pyplot as plt
	import tensorflow as tf
	from pandas import DataFrame,Series,read_csv
	
	from math import exp,sqrt,sin,pi,cos
	np.set_printoptions(precision=3)

		
	from Logger import *
	from RBM_Functions import *
	### import seaborn? ###
	if import_seaborn:
		import seaborn

		seaborn.set(font_scale=1.1)
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

		# plt.rcParams['image.cmap'] = 'coolwarm'
		# seaborn.set_palette(seaborn.color_palette("Set2", 10))

	log=Logger(True)

from tensorflow.examples.tutorials.mnist import input_data
time_now = time.asctime()
time_now = time_now.replace(":", "-")
time_now = time_now.replace(" ", "_")

#### Load T Data 
LOAD_MNIST = 1
LOAD_HORSES = 0
if "train_data" not in globals():
	if LOAD_MNIST:
		log.out("Loading Data")
		mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
		train_data, train_label, test_data, test_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
		#get test data of only one number class:
		index_for_number_test  = np.zeros([10,1200])
		where = np.zeros(10).astype(np.int)
		index_for_number_train = []
		for i in range(len(test_label)):
			for digit in range(10):
				d_array = np.zeros(10)
				d_array[digit] = 1
				if (test_label[i]==d_array).sum()==10:
					index_for_number_test[digit][where[digit]]=i
					where[digit]+=1
		index_for_number_test = index_for_number_test.astype(np.int)

		for i in range(len(train_label)):
			if (train_label[i]==[0,0,0,1,0,0,0,0,0,0]).sum()==10:
				index_for_number_train.append(i)

		# prepare  data for using on next RBM
		if 0:
			log.out("Making data for RBM")
			# where the weights of the beforehand trained RBM are stored
			bottom_w_file = "/Users/Niermann/Google Drive/Masterarbeit/Python/DBM Project/data/Sun_Apr__1_17-26-14_2018_[400, 100]/w0.txt"
			# where the bias of the beforehand trained RBM are stored (h layer)
			bottom_b_file = "/Users/Niermann/Google Drive/Masterarbeit/Python/DBM Project/data/Sun_Apr__1_17-26-14_2018_[400, 100]/bias1.txt"
			w_of_bottom_rbm = np.loadtxt(bottom_w_file)
			b_of_bottom_rbm = np.loadtxt(bottom_b_file)
			train_data = sigmoid_np(np.dot(train_data, w_of_bottom_rbm)+b_of_bottom_rbm,0.05)
			test_data = sigmoid_np(np.dot(test_data, w_of_bottom_rbm)+b_of_bottom_rbm,0.05)
	

			# 	new_data_train = np.zeros([len(train_label),100])
			# 	new_data_test = np.zeros([len(test_label),100])
			# 	for i in range(len(train_label)):
			# 		new_data_train[i]  = np.repeat(train_label[i],10).reshape(10,10).T.flatten()
			# 	for i in range(len(test_label)):
			# 		new_data_test[i]  = np.repeat(test_label[i],10).reshape(10,10).T.flatten()
			# 	train_label = new_data_train
			# 	test_label = new_data_test
			# log.out("Setting trian label = 10*10")
				# test_data_noise = np.copy(test_data)
				# # making noise 
				# for i in range(len(test_data_noise)):
				# 	test_data_noise[i]  += np.round(rnd.random(test_data_noise[i,:].shape)*0.55)
				# 	# half_images[i] = abs(half_images[i])
				# 	# half_images[i] *= 1/half_images[i].max()
				# 	# half_images[i] *= rnd.random(half_images[i].shape)
				# test_data_noise   = test_data_noise>0

				# noise_data_train = sample_np(rnd.random(train_data.shape)*0.2)
				# noise_data_test = sample_np(rnd.random(test_data.shape)*0.2)
				# noise_label_train = np.zeros(train_label.shape)
				# noise_label_test = np.zeros(test_label.shape)

	if LOAD_HORSES:
		log.out("Loading HORSE Data")
		horse_data_dir   = workdir+"/Horse_data_rescaled/"
		files      = os.listdir(horse_data_dir)
		train_data = np.zeros([len(files)-50,64**2])
		test_data  = np.zeros([50,64**2])

		from PIL import Image
		for i,f in enumerate(files):
			if f[-4:]==".jpg":
				img_data = np.array(Image.open(horse_data_dir+f)).flatten()/255.
				if i < train_data.shape[0]:
					train_data[i] = img_data
				else:
					test_data[i-train_data.shape[0]] = img_data
			else:
				log.info("Skipped %s"%f)

#### Get the arguments list from terminal
additional_args=sys.argv[1:]



################################################################################################################################################
### Class RBM 
class RBM(object):
	""" defines a 2 layer restricted boltzmann machine - first layer = input, second
	layer = output. Training with contrastive divergence """

	def __init__(self,vu,hu,forw_mult,back_mult,learnrate,liveplot=0):
		#### User Variables

		self.hidden_units  = hu
		self.visible_units = vu
		self.learnrate = learnrate


		self.liveplot  = liveplot

		self.forw_mult = forw_mult
		self.back_mult = back_mult


		################################################################################################################################################
		#### Graph
		################################################################################################################################################
		#shape definitions are wired here-normally one would define [rows,columns] but here it is reversed with [columns,rows]? """

		self.v       = tf.placeholder(tf.float32,[None,self.visible_units],name="Visible-Layer") 
		# has shape [number of images per batch,number of visible units]

		self.w       = tf.Variable(tf.random_uniform([self.visible_units,self.hidden_units],minval=-1e-6,maxval=1e-6),name="Weights")
		self.bias_v  = tf.Variable(tf.zeros([self.visible_units]),name="Visible-Bias")
		self.bias_h  = tf.Variable(tf.zeros([self.hidden_units]), name="Hidden-Bias")


		# get the probabilities of the hidden units in w
		self.h_prob  = sigmoid(tf.matmul(self.v,self.forw_mult*self.w) + self.bias_h,temp)
		# h has shape [number of images per batch, number of hidden units]
		# get the actual activations for h {0,1}
		# self.h       = tf.nn.relu(
		# 	            tf.sign(
		# 	            	self.h_prob - tf.random_uniform(tf.shape(self.h_prob)) 
		# 	            	) 
		#         		) 

		# and the same for visible units
		self.v_prob  = sigmoid(tf.matmul(self.h_prob,(self.back_mult*self.w),transpose_b=True) + self.bias_v,temp)
		self.v_recon = tf.nn.relu(
					tf.sign(
						self.v_prob - tf.random_uniform(tf.shape(self.v_prob))
						)
					)

		# Gibbs sampling: get the probabilities of h again from the reconstructed v_recon
		self.h_gibbs = sigmoid(tf.matmul(self.v_recon, self.w) + self.bias_h,temp) 

		##### define reconstruction error and the energy  
		# energy = -tf.reduce_sum(bias_v*v_recon)-tf.reduce_sum(bias_h*h)-tf.matmul(tf.matmul(h,tf.transpose(w)), v_recon)
		self.error  = tf.reduce_mean(tf.square(self.v-self.v_recon))

		#### Training with Contrastive Divergence
		#matrix shape is untouched throu the batches because w*v=h even if v has more columns, but dividing be numpoints is recomended since CD
		# [] = [784,batchsize]-transposed v * [batchsize,500] -> [784,500] - like w 
		self.pos_grad  = tf.matmul(self.v,self.h_prob,transpose_a=True)
		self.neg_grad  = tf.matmul(self.v_recon,self.h_gibbs,transpose_a=True)
		self.numpoints = tf.cast(tf.shape(self.v)[0],tf.float32) 
		#number of train inputs per batch (for averaging the CD matrix -> see practical paper by hinton)
		# contrastive divergence
		self.CD        = (self.pos_grad - self.neg_grad)/self.numpoints
		

		#update w
		self.update_w = self.w.assign_add(self.learnrate*self.CD)
		self.mean_w   = tf.reduce_mean(self.w)

		#update bias
		""" Since vectors v and h are actualy matrices with number of batch_size images in them, reduce mean will make them to a vector again """
		self.update_bias_v = self.bias_v.assign_add(self.learnrate*tf.reduce_mean(self.v-self.v_recon,0))
		self.update_bias_h = self.bias_h.assign_add(self.learnrate*tf.reduce_mean(self.h_prob-self.h_gibbs,0))


		# reverse feed
		# self.h_rev       = tf.placeholder(tf.float32,[None,self.hidden_units],name="Reverse-hidden")
		# self.v_prob_rev  = sigmoid(tf.matmul(self.h_rev,(self.w),transpose_b=True) + self.bias_v,temp)
		# self.v_recon_rev = tf.nn.relu(tf.sign(self.v_prob_rev - tf.random_uniform(tf.shape(self.v_prob_rev))))

	def train(self,sess,RBM_i,RBMs,batch):
		self.my_input_data = batch
		# iterate which RBM level this is and calculate the proper input 
		for j in range(1,len(RBMs)):
			if RBM_i >= j:
				self.my_input_data = RBMs[j-1].h_prob.eval({RBMs[j-1].v : self.my_input_data})

		#### update the weights and biases
		self.w_i, self.error_i = sess.run([self.update_w,self.error],feed_dict={self.v:self.my_input_data})
		sess.run([self.update_bias_h,self.update_bias_v],feed_dict={self.v:self.my_input_data})

		return self.w_i,self.error_i




################################################################################################################################################
### Class Deep BM 
class DBM_class(object):
	"""defines a deep boltzmann machine
	"""
	def __init__(self,shape,liveplot,classification):
		self.n_layers       = len(shape)
		self.liveplot       = liveplot # if true will open a lifeplot of the weight matrix 
		self.SHAPE          = shape  # contains the number of  neurons in a list from v layer to h1 to h2 
		self.classification = classification #weather the machine uses a label layer 
		
		self.init_state     = 0
		self.exported       = 0
		self.tested         = 0
		self.l_mean         = np.zeros([self.n_layers])
		


		self.train_time     = 0
		self.epochs         = 0
		self.recon_error_train = []
		self.class_error_train = []
		self.layer_diversity_ = []
		self.update = 0 	# update counter



		### save dictionary where time series data from test and train is stored
		self.save_dict ={	"Test_Epoch":    [],
							"Train_Epoch":   [],
							"Recon_Error":   [],
							"Class_Error":   [],
							"Temperature":   [],
							"Learnrate":     [],
							"Freerun_Steps": [],
							}
		for i in range(self.n_layers-1):
			self.save_dict["W_mean_%i"%i] = []
		for i in range(self.n_layers):
			self.save_dict["Layer_Activity_%i"%i] = []
			self.save_dict["Layer_Diversity_%i"%i] = []

		### log list where all constants are saved
		self.log_list =	[["SHAPE",                self.SHAPE],
						["N_EPOCHS_PRETRAIN",     N_EPOCHS_PRETRAIN], 
						["N_BATCHES_PRETRAIN",    N_BATCHES_PRETRAIN], 
						["N_BATCHES_TRAIN",       N_BATCHES_TRAIN], 
						["LEARNRATE_PRETRAIN",    LEARNRATE_PRETRAIN], 
						["LEARNRATE_START",       LEARNRATE_START], 
						["LEARNRATE_SLOPE",       LEARNRATE_SLOPE], 
						["TEMP_START",            TEMP_START], 
						["TEMP_SLOPE",            TEMP_SLOPE], 
						["PATHSUFFIX_PRETRAINED", PATHSUFFIX_PRETRAINED], 
						["PATHSUFFIX",            PATHSUFFIX], 
						["DO_LOAD_FROM_FILE",      DO_LOAD_FROM_FILE], 
						["TEST_EVERY_EPOCH",      TEST_EVERY_EPOCH]
						]## append variables that change during training in the write_to_file function


		log.out("Creating RBMs")
		self.RBMs    = [None]*(self.n_layers-1)
		for i in range(len(self.RBMs)):
			if i == 0 and len(self.RBMs)>1:
				self.RBMs[i] = RBM(self.SHAPE[i],self.SHAPE[i+1], forw_mult= 1, back_mult = 1, learnrate = LEARNRATE_PRETRAIN, liveplot=0)
				log.out("2,1")
			elif i==len(self.RBMs)-1 and len(self.RBMs)>1:
				self.RBMs[i] = RBM(self.SHAPE[i],self.SHAPE[i+1], forw_mult= 1, back_mult = 1, learnrate = LEARNRATE_PRETRAIN, liveplot=0)				
				log.out("1,2")
			else:
				if len(self.RBMs) == 1:
					self.RBMs[i] = RBM(self.SHAPE[i],self.SHAPE[i+1], forw_mult= 1, back_mult = 1, learnrate = LEARNRATE_PRETRAIN, liveplot=0)
					log.out("1,1")
				else:
					self.RBMs[i] = RBM(self.SHAPE[i],self.SHAPE[i+1], forw_mult= 1, back_mult = 1, learnrate = LEARNRATE_PRETRAIN, liveplot=0)
					log.out("2,2")
				
		# self.RBMs[1] = RBM(self.SHAPE[1],self.SHAPE[2], forw_mult= 1, back_mult = 1, learnrate = LEARNRATE_PRETRAIN, liveplot=0)
		# self.RBMs[2] = RBM(self.SHAPE[2],self.SHAPE[3], forw_mult= 1, back_mult = 1, learnrate = LEARNRATE_PRETRAIN, liveplot=0)

	def pretrain(self):
		""" this function will pretrain the RBMs and define a self.weights list where every
		weight will be stored in. This weights list can then be used to save to file and/or 
		to be loaded into the DBM for further training. 
		"""

		if DO_PRETRAINING:
			for rbm in self.RBMs:
				if rbm.liveplot:
					log.info("Liveplot is open!")
					fig,ax=plt.subplots(1,1,figsize=(15,10))
					break

		batchsize_pretrain = int(len(train_data)/N_BATCHES_PRETRAIN)

		with tf.Session() as sess:
			# train session - v has batchsize length
			log.start("Pretrain Session")
			
			
			#iterate through the RBMs , each iteration is a RBM
			if DO_PRETRAINING:	
				sess.run(tf.global_variables_initializer())

				for RBM_i, RBM in enumerate(self.RBMs):
					log.start("Pretraining ",str(RBM_i+1)+".", "RBM")
					

					for epoch in range(N_EPOCHS_PRETRAIN[RBM_i]):

						log.start("Epoch:",epoch+1,"/",N_EPOCHS_PRETRAIN[RBM_i])
						
						for start, end in zip( range(0, len(train_data), batchsize_pretrain), range(batchsize_pretrain, len(train_data), batchsize_pretrain)):
							#### define a batch
							batch = train_data[start:end]
							# train the rbm  
							w_i,error_i = RBM.train(sess,RBM_i,self.RBMs,batch)
							#### liveplot
							if RBM.liveplot and plt.fignum_exists(fig.number) and start%40==0:
								ax.cla()
								rbm_shape  = int(sqrt(RBM.visible_units))
								matrix_new = tile_raster_images(X=w_i.T, img_shape=(rbm_shape, rbm_shape), tile_shape=(10, 10), tile_spacing=(0,0))
								ax.matshow(matrix_new)
								plt.pause(0.00001)


						log.info("Learnrate:",round(LEARNRATE_PRETRAIN,4))
						log.info("error",round(error_i,4))
						log.end() #ending the epoch


					log.end() #ending training the rbm 

				

				# define the weights
				self.weights  =  []
				for i in range(len(self.RBMs)):
					self.weights.append(self.RBMs[i].w.eval())

				if DO_SAVE_PRETRAINED:
					for i in range(len(self.weights)):
						np.savetxt(workdir+"/pretrain_data/"+"Pretrained-"+" %i "%i+str(time_now)+".txt", self.weights[i])
					log.out("Saved Pretrained under "+str(time_now))
			else:
				if not DO_LOAD_FROM_FILE:
					### load the pretrained weights
					self.weights=[]
					log.out("Loading Pretrained from file")
					for i in range(self.n_layers-1):
						self.weights.append(np.loadtxt(workdir+"/pretrain_data/"+"Pretrained-"+" %i "%i+PATHSUFFIX_PRETRAINED+".txt").astype(np.float32))
				else:
					### if loading from file is active the pretrained weights would get 
					### reloaded anyway so directly load them here
					self.weights=[]
					log.out("Loading from file")
					for i in range(self.n_layers-1):
						self.weights.append(np.loadtxt(data_dir+"/"+PATHSUFFIX+"/"+"w%i.txt"%(i)).astype(np.float32))
			log.end()
			log.reset()

	def load_from_file(self,path,override_params=0):
		""" loads weights and biases from folder and sets 
		variables like learnrate and temperature to the values
		that were used in the last epoch"""
		global learnrate, temp, freerun_steps
		os.chdir(path)
		log.out("Loading data from:","...",path[-20:])

		self.w_np     = []
		self.w_np_old = []
		for i in range(self.n_layers-1):
			self.w_np.append(np.loadtxt("w%i.txt"%(i)))
			self.w_np_old.append(self.w_np[i])  #save weights for later comparison

		self.bias_np = []
		for i in range(self.n_layers):
			self.bias_np.append(np.loadtxt("bias%i.txt"%(i)))
		if override_params:
			try:
				log.out("Overriding Values from save")
				#
				sd = read_csv("save_dict.csv")
				l_ = sd["Learnrate"].values[[sd["Learnrate"].notnull()]]
				t_ = sd["Temperature"].values[[sd["Temperature"].notnull()]]
				n_ = sd["Freerun_Steps"].values[[sd["Freerun_Steps"].notnull()]]
				train_epoch_ = sd["Train_Epoch"].values[[sd["Train_Epoch"].notnull()]]

				freerun_steps = n_[-1]
				temp          = t_[-1]
				learnrate     = l_[-1]
				self.epochs   = train_epoch_[-1]

				log.info("Epoch = ",self.epochs)
				log.info("l = ",learnrate)
				log.info("T = ",temp)
				log.info("N = ",freerun_steps)
			except:
				log.error("Error overriding: Could not find save_dict.csv")
		os.chdir(workdir)

	def import_(self):
		""" setting up the graph and setting the weights and biases tf variables to the 
		saved numpy arrays """
		log.out("loading numpy vars into graph")
		for i in range(self.n_layers-1):
			sess.run(self.w[i].assign(self.w_np[i]))
		for i in range(self.n_layers):
			sess.run(self.bias[i].assign(self.bias_np[i]))

	def layer_input(self, layer_i):
		""" calculate input of layer layer_i
		layer_i :: for which layer
		returns :: input for the layer - which are the probabilites
		"""
		if layer_i == 0:
			_input_ = sigmoid(tf.matmul(self.layer[layer_i+1], self.w[layer_i],transpose_b=True) + self.bias[layer_i], self.temp_tf)			

		elif layer_i == self.n_layers-1:
			_input_ = sigmoid(tf.matmul(self.layer[layer_i-1],self.w[layer_i-1]) + self.bias[layer_i], self.temp_tf)
		
		else:
			_input_ = sigmoid(tf.matmul(self.layer[layer_i-1],self.w[layer_i-1]) 
					+ tf.matmul(self.layer[layer_i+1],self.w[layer_i],transpose_b=True) 
					+ self.bias[layer_i], 
					self.temp_tf)
		return _input_

	def sample(self,x):
		""" takes sample from x where x is a probability vector.
		subtracts a random uniform from each prob and then applies the 
		sign function to just get +1,-1 , then a relu is applied to set 
		every elelemt with negative sign to 0
		"""
		return tf.nn.relu(
				tf.sign(
					x - tf.random_uniform(tf.shape(x))
					)
				) 

	def get_learnrate(self,epoch,a,y_off):
		""" calculate the learnrate dependend on parameters
		epoch :: current epoch 
		a :: slope
		y_off :: y offset -> start learningrate
		"""
		L = a / (float(a)/y_off+epoch)
		return L

	def get_temp(self,epoch,a,y_off):
		""" calculate the Temperature dependend on parameters 
		epoch :: current epoch 
		a :: slope
		y_off :: y offset -> start learningrate
		"""
		T = a / (float(a)/y_off+epoch)
		return T

	def get_N(self,epoch):
		N = 2
		return N

	def update_savedict(self,mode):
		if mode=="training":
			# append all data to save_dict
			self.save_dict["Train_Epoch"].append(self.epochs)
			self.save_dict["Temperature"].append(temp)
			self.save_dict["Learnrate"].append(learnrate)
			self.save_dict["Freerun_Steps"].append(freerun_steps)
			
			for i in range(self.n_layers-1):
				w_mean = sess.run(self.mean_w)
				self.save_dict["W_mean_%i"%i].append(w_mean)

			for i in range(self.n_layers):
				self.save_dict["Layer_Activity_%i"%i].append(self.l_mean[i])
				self.save_dict["Layer_Diversity_%i"%i].append(self.layer_diversity_[-1][i])


		if mode == "testing":

			if self.classification:
				self.save_dict["Class_Error"].append(self.class_error_test)

			self.save_dict["Recon_Error"].append(self.recon_error)
			self.save_dict["Test_Epoch"].append(self.epochs)

	def graph_init(self,graph_mode):
		""" sets the graph up and loads the pretrained weights in , these are given
		at class definition
		graph_mode  :: "training" if the graph is used in training - this will set h2 to placeholder for the label data
				:: "testing"  if the graph is used in testing - this will set h2 to a random value and to be calculated from h1 
				:: "gibbs"    if the graph is used in gibbs sampling 
		"""
		log.out("Initializing graph")
		

		# self.v  = tf.placeholder(tf.float32,[self.batchsize,self.SHAPE[0]],name="Visible-Layer") 

		if graph_mode=="training":
			# stuff
			# self.m_tf      = tf.placeholder(tf.int32,[],name="running_array_index")
			self.learnrate = tf.placeholder(tf.float32,[],name="Learnrate")

			# arrays for saving progress
			# self.h1_activity_ = tf.Variable(tf.zeros([N_EPOCHS_TRAIN]))
			# self.h2_activity_ = tf.Variable(tf.zeros([N_EPOCHS_TRAIN]))
			# self.train_error_ = tf.Variable(tf.zeros([N_EPOCHS_TRAIN]))
			# self.train_class_error_ = tf.Variable(tf.zeros([N_EPOCHS_TRAIN]))
			
		#### temperature
		self.temp_tf = tf.placeholder(tf.float32,[],name="Temperature")


		### init all Parameters like weights , biases , layers and their updates
		# weights
		self.w               = [None]*(self.n_layers-1)
		self.pos_grad        = [None]*(self.n_layers-1)
		self.neg_grad        = [None]*(self.n_layers-1)
		self.update_pos_grad = [None]*(self.n_layers-1)
		self.update_neg_grad = [None]*(self.n_layers-1)
		self.update_w        = [None]*(self.n_layers-1)
		self.w_mean_         = [None]*(self.n_layers-1) # variable to store means
		self.mean_w          = [None]*(self.n_layers-1) # calc of mean for each w
		self.do_norm_w       = [None]*(self.n_layers-1)
		# bias
		self.bias        = [None]*self.n_layers
		self.update_bias = [None]*self.n_layers

		# layer
		self.layer             = [None]*self.n_layers # layer variable 
		self.layer_save        = [None]*self.n_layers # save variable (used for storing older layers)
		self.assign_save_layer = [None]*self.n_layers # save variable (used for storing older layers)
		self.layer_ph          = [None]*self.n_layers # placeholder 
		self.assign_l          = [None]*self.n_layers # assign op. (assigns placeholder)
		self.assign_l_rand     = [None]*self.n_layers # assign op. (assigns random)
		self.layer_prob        = [None]*self.n_layers # calc prob for layer n
		self.layer_samp        = [None]*self.n_layers # take a sample from the prob
		self.update_l_s        = [None]*self.n_layers # assign op. for calculated samples
		self.update_l_p        = [None]*self.n_layers # assign op. for calculated probs
		self.layer_activities  = [None]*self.n_layers # calc for layer activieties (mean over batch)
		self.layer_energy      = [None]*(self.n_layers-1)
		self.layer_diversity   = [None]*self.n_layers # measure how diverse each layer is in the batch 


		### layer vars 
		for i in range(len(self.layer)):
			self.layer[i]      = tf.Variable(tf.random_uniform([self.batchsize,self.SHAPE[i]],minval=-1e-3,maxval=1e-3),name="Layer_%i"%i)
			self.layer_save[i] = tf.Variable(tf.random_uniform([self.batchsize,self.SHAPE[i]],minval=-1e-3,maxval=1e-3),name="Layer_save_%i"%i)
			self.layer_ph[i]   = tf.placeholder(tf.float32,[self.batchsize,self.SHAPE[i]],name="layer_%i_PH"%i)

		### weight calculations and assignments
		for i in range(len(self.w)):
			self.w[i] = tf.Variable(self.weights[i],name="Weights%i"%i)
			if graph_mode=="training":
				self.pos_grad[i]        = tf.Variable(tf.zeros([self.SHAPE[i],self.SHAPE[i+1]]))
				self.neg_grad[i]        = tf.Variable(tf.zeros([self.SHAPE[i],self.SHAPE[i+1]]))
				self.update_pos_grad[i] = self.pos_grad[i].assign(tf.matmul(self.layer[i], self.layer[i+1],transpose_a=True))
				self.update_neg_grad[i] = self.neg_grad[i].assign(tf.matmul(self.layer[i], self.layer[i+1],transpose_a=True))
				self.update_w[i]        = self.w[i].assign_add(self.learnrate*(self.pos_grad[i] - self.neg_grad[i])/self.batchsize)
				self.w_mean_[i]         = tf.Variable(tf.zeros([N_EPOCHS_TRAIN]))
				self.mean_w[i]          = tf.sqrt(tf.reduce_sum(tf.square(self.w[i])))
				self.do_norm_w[i]       = self.w[i].assign(self.w[i]/tf.sqrt(tf.reduce_sum(tf.square(self.w[i]))))


		### bias calculations and assignments
		for i in range(len(self.bias)):
			self.bias[i] = tf.Variable(tf.zeros([self.SHAPE[i]]),name="Bias%i"%i)
			if graph_mode == "training":
				self.update_bias[i] = self.bias[i].assign_add(self.learnrate*tf.reduce_mean(tf.subtract(self.layer_save[i],self.layer[i]),0))

		### layer calculations and assignments
		for i in range(len(self.layer)):
			self.assign_save_layer[i]       = self.layer_save[i].assign(self.layer[i])
			self.assign_l[i]         = self.layer[i].assign(self.layer_ph[i])
			self.assign_l_rand[i]    = self.layer[i].assign(tf.random_uniform([self.batchsize,self.SHAPE[i]]))
			self.layer_prob[i]       = self.layer_input(i)
			self.layer_samp[i]       = self.sample(self.layer_prob[i])
			self.update_l_p[i]       = self.layer[i].assign(self.layer_prob[i])
			self.layer_activities[i] = tf.reduce_sum(self.layer[i])/(self.batchsize*self.SHAPE[i])*100
			self.layer_diversity[i]  = tf.reduce_mean(tf.abs(self.layer[i] - tf.reduce_mean(self.layer[i], axis=0)))

		for i in range(len(self.layer)-1):
			self.layer_energy[i] = tf.matmul(self.layer[i], tf.matmul(self.w[i],self.layer[i+1],transpose_b=True))
			self.update_l_s[i]   = self.layer[i].assign(self.layer_samp[i])
		self.update_l_s[-1] = self.layer[-1].assign(self.layer_prob[-1])

		# modification array size 10 that gehts multiplied to the label vector for context
		self.modification_tf = tf.Variable(tf.ones([self.batchsize,self.SHAPE[-1]]),name="Modification")



		### Error and stuff
		self.error       = tf.reduce_mean(tf.square(self.layer_ph[0]-self.layer[0]))
		self.class_error = tf.reduce_mean(tf.square(self.layer_ph[-1]-self.layer[-1]))

		self.h1_sum    = tf.reduce_sum(self.layer[1])
		# self.h2_sum    = tf.reduce_sum(self.layer[2])
		self.label_sum = tf.reduce_sum(self.layer[-1])

		self.free_energy = -tf.reduce_sum(tf.log(1+tf.exp(tf.matmul(self.layer[0],self.w[0])+self.bias[1])))

		self.energy = -tf.reduce_sum([self.layer_energy[i] for i in range(len(self.layer_energy))])

		#### updates for each layer 
		if self.classification and self.n_layers > 2:
			self.update_h2_with_context = self.layer[-2].assign(self.sample(sigmoid(tf.matmul(self.layer[-3],self.w[-2])  
								+ tf.matmul(tf.multiply(self.layer[-1],self.modification_tf),self.w[-1],transpose_b=True)
								+ self.bias[-2],self.temp_tf)))
		

		### Training with contrastive Divergence
		# if graph_mode=="training":
			# self.assign_arrays =	[ tf.scatter_update(self.train_error_, self.m_tf, self.error), 							  
			# 				  tf.scatter_update(self.h1_activity_, self.m_tf, self.h1_sum), 
			# 				]

			# for i in range(self.n_layers-1):
			# 	self.assign_arrays.append(tf.scatter_update(self.w_mean_[i], self.m_tf, self.mean_w[i]))
			# if self.classification:
			# 	self.assign_arrays.append(tf.scatter_update(self.train_class_error_, self.m_tf, self.class_error))

		sess.run(tf.global_variables_initializer())
		self.init_state=1

	def test_noise_stability(self,input_data,input_label,steps):
		self.batchsize=len(input_data)
		if DO_LOAD_FROM_FILE:
			self.load_from_file(workdir+"/data/"+PATHSUFFIX)
		self.graph_init("testing")
		self.import_()

		n       = 20
		h2_     = []
		r       = rnd.random([self.batchsize,784])
		v_noise = np.copy(input_data)
		# make the input more noisy
		v_noise += (abs(r-0.5)*0.5)
		v_noise = sample_np(v_noise)

		sess.run(self.assign_l[0] , {self.layer_ph[0] : v_noise})
		sess.run(self.update_l_p[1], {self.temp_tf : temp})
		sess.run(self.update_l_p[2], {self.temp_tf : temp})

		for i in range(steps):
			
			layer = sess.run(self.update_l_s, {self.temp_tf : temp})
			

			if self.classification:
				h2_.append(layer[-1])


		v_noise_recon = sess.run(self.update_l_p[0], {self.temp_tf : temp})
		return np.array(h2_),v_noise_recon,v_noise

	def train(self,train_data,train_label,num_batches,cont):
		global learnrate, temp, freerun_steps
		""" training the DBM with given h2 as labels and v as input images
		train_data  :: images
		train_label :: corresponding label
		num_batches :: how many batches
		"""
		######## init all vars for training
		self.batchsize = int(len(train_data)/num_batches)
		self.num_of_updates = N_EPOCHS_TRAIN*num_batches


		# number of clamped sample steps
		if self.n_layers <=3 and self.classification==1:
			M = 2
		else:
			M = 10




		### free energy
		# self.F=[]
		# self.F_test=[]
		
		if DO_LOAD_FROM_FILE and not cont:
			# load data from the file
			self.load_from_file(workdir+"/data/"+PATHSUFFIX,override_params=1)
			self.graph_init("training")
			self.import_()

		# if no files loaded then init the graph with pretrained vars
		if self.init_state==0:
			self.graph_init("training")

		if cont and self.tested:
			self.graph_init("training")
			self.import_()
			self.tested = 0


		if self.liveplot:
			log.info("Liveplot is on!")
			fig,ax = plt.subplots(1,1,figsize=(15,10))
			data   = ax.matshow(tile(self.w[0].eval()), vmin=-0.01, vmax=0.01)
			plt.colorbar(data)


		# starting the training
		log.info("Batchsize:",self.batchsize,"N_Updates",self.num_of_updates)
		
		log.start("Deep BM Epoch:",self.epochs+1,"/",N_EPOCHS_TRAIN)

		# shuffle test data and labels so that batches are not equal every epoch 
		log.out("Shuffling TrainData")
		self.seed   = rnd.randint(len(train_data),size=(int(len(train_data)/10),2))
		train_data  = shuffle(train_data, self.seed)
		if self.classification:
			train_label = shuffle(train_label, self.seed)

		log.out("Running Batch")
		# log.info("++ Using Weight Decay! Not updating bias! ++")


		for start, end in zip( range(0, len(train_data), self.batchsize), range(self.batchsize, len(train_data), self.batchsize)):
			# define a batch
			batch = train_data[start:end]
			if self.classification:
				batch_label = train_label[start:end]

			#### Clamped Run 
			# assign v and h2 to the batch data
			sess.run(self.assign_l[0], { self.layer_ph[0]  : batch })
			if self.classification:
				sess.run(self.assign_l[-1], {self.layer_ph[-1] : batch_label})

			# calc hidden layer probabilities (not the visible & label layer)
			for hidden in range(M):
				if self.classification:
					sess.run(self.update_l_s[1:-1],{self.temp_tf : temp})
				else:
					sess.run(self.update_l_s[1:],{self.temp_tf : temp})

			# last run calc only the probs to reduce noise
			sess.run(self.update_l_p[1:-1],{self.temp_tf : temp})
			# save all layer for bias update
			sess.run(self.assign_save_layer)
			# update the positive gradients
			sess.run(self.update_pos_grad)

			


			#### Free Running 
			# update all layers N times (Gibbs sampling) 
			for n in range(freerun_steps):
				# using sampling
				sess.run(self.update_l_s,{self.temp_tf : temp})
			sess.run(self.update_l_p,{self.temp_tf : temp})
			# calc he negatie gradients
			sess.run(self.update_neg_grad)



			self.layer_diversity_.append(sess.run(self.layer_diversity))



			#### run all parameter updates 
			sess.run([self.update_w, self.update_bias], {self.learnrate : learnrate})
			## norm the weights
			sess.run(self.do_norm_w)

			### calc errors 
			self.recon_error_train.append(sess.run(self.error,{self.layer_ph[0] : batch}))
			if self.classification:
				self.class_error_train.append(sess.run(self.class_error,{self.layer_ph[-1] : batch_label}))

			#### calculate free energy for test and train data
			# self.F.append(self.free_energy.eval({self.v:self.batch}))
			# f_test_ = self.free_energy.eval({self.v:test_data[0:self.batchsize]})
			# for i in range(1,10):
			# 	f_test_ += self.free_energy.eval({self.v:test_data[i*self.batchsize:i*self.batchsize+self.batchsize]})
			# f_test_*=1./10
			# self.F_test.append(f_test_)

			self.l_mean += sess.run(self.layer_activities)

			self.update += 1

			### liveplot
			if self.liveplot and plt.fignum_exists(fig.number) and start%40==0:
				if start%4000==0:
					ax.cla()
					data = ax.matshow(tile(self.w[0].eval()),vmin=tile(self.w[0].eval()).min()*1.2,vmax=tile(self.w[0].eval()).max()*1.2)

				matrix_new = tile(self.w[0].eval())
				data.set_data(matrix_new)
				plt.pause(0.00001)

		
		log.end() #ending the epoch

		### write vars into savedict
		self.update_savedict("training")
		self.l_mean[:] = 0

		# increase epoch counter
		self.epochs += 1 
		
		# change learnrate
		log.info("Learnrate: ",np.round(learnrate,5))
		learnrate = self.get_learnrate(self.epochs, LEARNRATE_SLOPE, LEARNRATE_START)
		
		# change temo
		log.info("Temp: ",np.round(temp,5))
		temp = self.get_temp(self.epochs, TEMP_SLOPE, TEMP_START)

		# change freerun_steps
		log.info("freerun_steps: ",freerun_steps)
		freerun_steps = self.get_N(self.epochs)

		# average layer activities over epochs 
		self.l_mean *= 1.0/num_batches

		self.export()

		log.reset()

	def test(self,my_test_data,my_test_label,N,M,create_conf_mat):
		""" testing runs without giving h2 , only v is given and h2 has to be infered 
		by the DBM 
		array my_test_data :: images to test, get assigned to v layer
		int N :: Number of updates from hidden layers 
		int M :: Number of samples taken to reconstruct the image
		"""
		#init the vars and reset the weights and biases 		
		self.batchsize=len(my_test_data)
		self.learnrate = LEARNRATE_START

		# h1    = np.zeros([N,self.batchsize,self.SHAPE[1]])
		# h2    = np.zeros([N,self.batchsize,self.SHAPE[2]])
		# label = np.zeros([N,self.batchsize,self.SHAPE[-1]])

		self.label_diff = np.zeros([N,self.batchsize,self.SHAPE[-1]])


		### init the graph 
		if DO_LOAD_FROM_FILE and not DO_TRAINING:
			self.load_from_file(workdir+"/data/"+PATHSUFFIX,override_params=1)
		self.graph_init("testing") # "testing" because this graph creates the testing variables where only v is given, not h2
		self.import_()


		#### start test run
		log.start("Testing DBM with %i images"%self.batchsize)

		#### give input to v layer
		sess.run(self.assign_l[0], {self.layer_ph[0] : my_test_data, self.temp_tf : temp})

		#### update hidden and label N times
		log.out("Sampling hidden %i times "%N)
		self.layer_act = np.zeros([N,self.n_layers])
		self.save_h1 = []
		
		for n in range(N):
			self.layer_act[n,:] = sess.run(self.layer_activities, {self.temp_tf : temp})
			self.hidden_save    = sess.run([self.update_l_p[i] for i in range(1,self.n_layers)], {self.temp_tf : temp})
			sess.run(self.update_l_p[1:],{self.temp_tf : temp})
			# self.save_h1.append(self.layer[1].eval()[0])
		

		
		self.h1_test = self.hidden_save[0]
		self.last_layer_save = self.layer[-1].eval()

		#### update v M times
		self.probs = self.layer[0].eval()
		self.image_timeline = []
		for i in range(M):
			self.probs += sess.run(self.update_l_s[0],{self.temp_tf : temp})
			self.image_timeline.append(self.layer[0].eval()[0])
		self.probs *= 1./(M+1)


		## check how well the v layer can reconstruct the h1 
		if self.n_layers == 2:
			__v__ = sigmoid_np(np.dot(self.h1_test, self.w_np[0].T)+self.bias_np[0],temp)
			self.h_reverse_recon = sigmoid_np(np.dot(__v__, self.w_np[0])+self.bias_np[1],temp)
			self.recon_error_reverse = np.mean(np.square(self.h1_test-self.h_reverse_recon))

		#### calculate errors and activations
		self.recon_error  = self.error.eval({self.layer_ph[0] : my_test_data})
		

		#### count how many images got classified wrong 
		log.out("Taking only the maximum")
		n_wrongs             = 0
		# label_copy         = np.copy(self.last_layer_save)
		wrong_classified_ind = []
		wrong_maxis          = []
		right_maxis          = []
		

		if self.classification:
			## error of classifivation labels
			self.class_error_test=np.mean(np.abs(self.last_layer_save-my_test_label[:,:10]))		
			
			for i in range(len(self.last_layer_save)):
				digit   = np.where(my_test_label[i]==1)[0][0]
				maxi    = self.last_layer_save[i].max()
				max_pos = np.where(self.last_layer_save[i] == maxi)[0][0]
				if max_pos != digit:
					wrong_classified_ind.append(i)
					wrong_maxis.append(maxi)#
				elif max_pos == digit:
					right_maxis.append(maxi)
			n_wrongs = len(wrong_maxis)

			if create_conf_mat:
				log.out("Making Confusion Matrix")
				
				self.conf_data = np.zeros([10,1,10]).tolist()

				for i in range(self.batchsize):
					digit = np.where( test_label[i] == 1 )[0][0]
					
					self.conf_data[digit].append( self.last_layer_save[i].tolist() )

				# confusion matrix
				w = np.zeros([10,10])
				for digit in range(10):
					w[digit]  = np.round(np.mean(np.array(DBM.conf_data[digit]),axis=0),3)
				seaborn.heatmap(w*100,annot=True)
				plt.ylabel("Desired Label in %")
				plt.xlabel("Predicted Label in %")
								
		self.class_error_test = float(n_wrongs)/self.batchsize

			# self.class_error_.append(float(n_wrongs)/self.batchsize)
			# self.test_epochs.append(self.epochs)
			# self.test_error_.append(self.recon_error) #append to errors if called multiple times
		
		# append test results to save_dict
		self.update_savedict("testing")

		self.tested = 1 # this tells the train function that the batchsize has changed 
		
		log.end()
		log.info("------------- Test Log -------------")
		log.info("Reconstr. error normal: ",np.round(self.recon_error,5))
		if self.n_layers==2: log.info("Reconstr. error reverse: ",np.round(self.recon_error_reverse,5)) 
		if self.classification:
			log.info("Class error: ",np.round(self.class_error_test, 5))
			log.info("Wrong Digits: ",n_wrongs," with average: ",round(np.mean(wrong_maxis),3))
			log.info("Correct Digits: ",len(right_maxis)," with average: ",round(np.mean(right_maxis),3))
		log.reset()
		return wrong_classified_ind

	def gibbs_sampling(self,v_input,gibbs_steps,TEMP_START,temp_end,subspace,mode,liveplot=1):
		""" Repeatedly samples v and label , where label can be modified by the user with the multiplication
		by the modification array - clamping the labels to certain numbers.
		v_input :: starting with an image as input can also be a batch of images
		
		temp_end, TEMP_START :: temperature will decrease or increase to temp_end and start at TEMP_START 
		
		mode 	:: "sampling" calculates h2 and v back and forth usign previous steps
			:: "context" clamps v and only calculates h1 based on previous h2
		
		subspace :: {"all", array} if "all" do nothing, if array: set the weights to 0 for all indices not marked by subspace
		 		used with "context" mode for clamping certain labels to 0
		"""

		self.layer_save = []
		for i in range(self.n_layers):
			self.layer_save.append(np.zeros([gibbs_steps,self.batchsize,self.SHAPE[i]]))

		temp_          = np.zeros([gibbs_steps])
		self.energy_   = []
		self.mean_h1   = []
		temp           = TEMP_START
		temp_delta     = (temp_end-TEMP_START)/gibbs_steps

		self.num_of_updates = 1000 #just needs to be defined because it will make a train graph with tf.arrays where this number is needed


		if liveplot:
			log.info("Liveplotting gibbs sampling")
			fig,ax=plt.subplots(1,self.n_layers+1,figsize=(15,6))
			# plt.tight_layout()

		log.start("Gibbs Sampling")
		log.info(": Mode: %s | Steps: %i"%(mode,gibbs_steps))

		if mode=="context":
			sess.run(self.assign_l[0],{self.layer_ph[0] : v_input})
			for i in range(1,self.n_layers):
				sess.run( self.assign_l[i], {self.layer_ph[i] : 0.01*rnd.random([self.batchsize, self.SHAPE[i]])} )
			
			# modification = np.concatenate((modification,)*self.batchsize).reshape(self.batchsize,10)
			# sess.run(self.modification_tf.assign(modification))


			# set the weights to 0 if context is enebaled and subspace is not "all"
			if subspace == "all":
				pass
			else:
				# get all numbers that are not in subspace
				subspace_anti = []
				for i in range(10):
					if i not in subspace:
						subspace_anti.append(i)
				

				log.out("Setting Weights to 0")
				# get the weights as numpy arrays
				w_ = self.w[-1].eval()
				b_ = self.bias[-1].eval()
				# set values to 0
				w_[:,subspace_anti] = 0
				b_[subspace_anti] = -10
				# assign to tf variables
				sess.run(self.w[-1].assign(w_))
				sess.run(self.bias[-1].assign(b_))

			### gibbs steps
			for step in range(gibbs_steps):
				# update all layer except first one
				layer = sess.run(self.update_l_s[1:], {self.temp_tf : temp})
				# layer_2 = sess.run(self.update_h2_with_context, {self.temp_tf : temp})
				# layer_3 = sess.run(self.update_l_s[-1], {self.temp_tf : temp})

				### save a generated image 
				# self.v_g = sigmoid_np(np.dot(self.w_np[0],layer[0][0])+self.bias_np[0], temp)

				# save layers 
				if liveplot:
					self.layer_save[0][step] = self.layer[0].eval()
					for layer_i in range(1,self.n_layers-2):
						self.layer_save[layer_i][step] = layer[layer_i]
					self.layer_save[-2][step] = layer[-2]
					# calc the energy
					self.energy_.append(sess.run(self.energy))
					# save values to array
					temp_[step] = temp

				self.layer_save[-1][step] = layer[-1]
				

				# assign new temp
				temp += temp_delta 


			#### calc the probabiliy for every point in h 
			# self.p_h = sigmoid_np(np.dot(h2,self.w2_np.T)+np.dot(v_gibbs,self.w[0]_np), temp)
			# # calc the standart deviation for every point
			# self.std_dev_h = np.sqrt(self.p_h*(1-self.p_h))

			#### for checking of the thermal equilibrium
			# step=10
			# if i%step==0 and i>0:
			# 	self.mean_h1.append( np.mean(h1_[i-(step-1):i], axis = (0,1) ))
			# 	if len(self.mean_h1)>1:
			# 		log.out(np.mean(abs(self.std_dev_h-(abs(self.mean_h1[-2]-self.mean_h1[-1])))))
	

		if mode=="generate":
			sess.run(self.assign_l_rand)
			sess.run(self.layer[-1].assign(v_input))

			for step in range(gibbs_steps):
				# update all layer except the last one 
				layer = sess.run(self.update_l_s[:-1], {self.temp_tf : temp})


				# save layers 
				if liveplot:
					for layer_i in range(len(layer)):
						self.layer_save[layer_i][step] = layer[layer_i]
					self.layer_save[-1][step] = v_input
					# calc the energy
					self.energy_.append(sess.run(self.energy))
					# save values to array
					temp_[step] = temp
				

				# assign new temp
				temp += temp_delta 
		
		if mode=="freerunning":
			sess.run(self.assign_l_rand)
			rng  =  rnd.randint(100)
			sess.run(self.assign_l[0], {self.layer_ph[0] : test_data[rng:rng+1]})
			for i in range(10):
				sess.run(self.update_l_s[1:],{self.temp_tf : temp})

			for step in range(gibbs_steps):
				
				# update all layer 
				# layer = [None]*self.n_layers
				layer=sess.run(self.update_l_s, {self.temp_tf : temp}) 
				# layer[1:]=sess.run(self.update_l_s[1:], {self.temp_tf : temp})
				
				
				if liveplot:
					# ass_save_layer
					for layer_i in range(len(layer)):
						self.layer_save[layer_i][step] = layer[layer_i]
					# calc the energy
					self.energy_.append(sess.run(self.energy))
					# save values to array
					temp_[step] = temp
			
				# assign new temp
				temp += temp_delta 

	
		if liveplot and plt.fignum_exists(fig.number) and self.batchsize==1:
			data = [None]*(self.n_layers+1)
			ax[0].set_title("Visible Layer")
			for layer_i in range(len(self.layer_save)):
				s = int(sqrt(self.SHAPE[layer_i]))
				if s!=3:
					data[layer_i]  = ax[layer_i].matshow(self.layer_save[layer_i][0].reshape(s,s),vmin=0,vmax=1)
					ax[layer_i].set_xticks([])
					ax[layer_i].set_yticks([])

				# ax[layer_i].set_yticks([])
				# ax[layer_i].set_xticks([])
				# ax[layer_i].grid(False)
			
			if self.classification:
				data[-2], = ax[-2].plot([],[])
				ax[-2].set_ylim(0,1)
				ax[-2].set_xlim(0,10)
				ax[-2].set_title("Classification")

			data[-1], = ax[-1].plot([],[])
			ax[-1].set_xlim(0,len(self.energy_))
			ax[-1].set_ylim(np.min(self.energy_),0)
			ax[-1].set_title("Energy")

			
			for step in range(1,gibbs_steps-1,6):
				if plt.fignum_exists(fig.number):
					ax[1].set_title("Temp.: %s, Steps: %s"%(str(round(temp_[step],3)),str(step)))

					for layer_i in range(len(self.layer_save)):
						s = int(sqrt(self.SHAPE[layer_i]))
						if s!=3:
							data[layer_i].set_data(self.layer_save[layer_i][step].reshape(s,s))
					if self.classification:
						data[-2].set_data(range(10),self.layer_save[-1][step])
					data[-1].set_data(range(step),self.energy_[:step])
					
					plt.pause(1/50.)
		
			plt.close(fig)

		log.end()
		if mode=="freerunning" or mode=="generate":
			# return the last images that got generated 
			v_layer = sess.run(self.update_l_p[0], {self.temp_tf : temp})
			return v_layer

		else:
			# return the mean of the last 20 gibbs samples for all images
			return np.mean(self.layer_save[-1][-20:,:],axis=0)

	def export(self):
		# convert weights and biases to numpy arrays
		self.w_np=[]
		for i in range(self.n_layers-1):
			self.w_np.append(self.w[i].eval())
		self.bias_np = []
		for i in range(self.n_layers):	
			self.bias_np.append(self.bias[i].eval())

		# convert tf.arrays to numpy arrays 
		# if training:
		# 	self.h1_activity_np = self.h1_activity_.eval()
		# 	self.h2_activity_np = self.h2_activity_.eval()
		# 	self.train_error_np = self.train_error_.eval()
		# 	self.train_class_error_np = self.train_class_error_.eval()
		# 	self.w_mean_np = []
		# 	for i in range(self.n_layers-1):
		# 		self.w_mean_np.append(self.w_mean_[i].eval())
		
		self.exported = 1

	def write_to_file(self):
		if self.exported!=1:
			self.export()
		new_path = saveto_path
		if not os.path.isdir(saveto_path):
			os.makedirs(new_path)
		os.chdir(new_path)
		
		# save weights 
		for i in range(self.n_layers-1):
			np.savetxt("w%i.txt"%i, self.w_np[i])
		
		##  save bias
		for i in range(self.n_layers):
			np.savetxt("bias%i.txt"%i, self.bias_np[i])
		
		## save log
		self.log_list.append(["train_time",self.train_time])
		self.log_list.append(["Epochs",self.epochs])
		
		## save save_dict
		try:
			save_df = DataFrame(dict([ (k,Series(v)) for k,v in self.save_dict.iteritems() ]))
		except:
			log.out("using dataframe items conversion for python 3.x")
			save_df = DataFrame(dict([ (k,Series(v)) for k,v in self.save_dict.items() ]))
		save_df.to_csv("save_dict.csv")

		## logfile
		with open("logfile.txt","w") as log_file:
				for i in range(len(self.log_list)):
					log_file.write(self.log_list[i][0]+","+str(self.log_list[i][1])+"\n")

		log.info("Saved data and log to:",new_path)
		os.chdir(workdir)


###########################################################################################################
#### User Settings ###

N_BATCHES_PRETRAIN = 300 			# how many batches per epoch for pretraining
N_BATCHES_TRAIN    = 300 			# how many batches per epoch for complete DBM training
N_EPOCHS_PRETRAIN  = [10,0,0,0,0,0] 	# pretrain epochs for each RBM
N_EPOCHS_TRAIN     = 4				# how often to iter through the test images
TEST_EVERY_EPOCH   = 5  			# how many epochs to train before testing on the test data

### learnrates 
LEARNRATE_PRETRAIN = 0.01		# learnrate for pretraining
LEARNRATE_START    = 0.01		# starting learnrates
LEARNRATE_SLOPE    = 0.1		# bigger number -> smaller slope

### temperature
TEMP_START    = 0.1 		# starting temp
TEMP_SLOPE    = 5000.0		# slope of dereasing temp bigger number -> smaller slope


### state vars 
DO_PRETRAINING = 1		# if no pretrain then files are automatically loaded
DO_TRAINING    = 1		# if to train the whole DBM
DO_TESTING     = 1		# if testing the DBM with test data
DO_SHOW_PLOTS  = 1		# if plots will show on display - either way they get saved into saveto_path

DO_CONTEXT    = 0	 	# if to test the context
DO_GEN_IMAGES = 0	 	# if to generate images (mode can be choosen at function call)
DO_NOISE_STAB = 0	 	# if to make a noise stability test


### saving and loading 
DO_SAVE_TO_FILE       = 0 	# if to save plots and data to file
DO_SAVE_PRETRAINED    = 0 	# if to save the pretrained weights seperately (for later use)
DO_LOAD_FROM_FILE     = 0 	# if to load weights and biases from datadir + pathsuffix
PATHSUFFIX            = "Wed_May_16_18-56-19_2018_[784, 144, 100, 25, 10]"
PATHSUFFIX_PRETRAINED = "Fri_Mar__9_16-46-01_2018"


DBM_SHAPE = [	int(sqrt(len(train_data[0])))*int(sqrt(len(train_data[0]))),
				12*12,
				10]
###########################################################################################################

### globals (to be set as DBM self values)
freerun_steps = 2 					# global number of freerun steps for training
learnrate     = LEARNRATE_START		# global learnrate
temp          = TEMP_START			# global temp state


saveto_path=data_dir+"/"+time_now+"_"+str(DBM_SHAPE)

### modify the parameters with additional_args
if len(additional_args) > 0:
	# t             = [1.5, 2, 2.5, 3, 3.5, 4, 4.5]
	# l             = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
	# temp          = t[int(additional_args[1])]
	# TEMP_START    = temp
	# LEARNRATE_START = l[int(additional_args[0])]
	saveto_path  += " - " + str(additional_args)

## open the logger-file
if DO_TRAINING and DO_SAVE_TO_FILE:
	os.makedirs(saveto_path)
	log.open(saveto_path)


######### DBM #############################################################################################
DBM = DBM_class(	shape = DBM_SHAPE,
					liveplot = 0, 
					classification = 1,
			)

###########################################################################################################
#### Sessions ####
log.reset()
log.info(time_now)


DBM.pretrain()

if DO_TRAINING:
	log.start("DBM Train Session")
	

	with tf.Session() as sess:

		for run in range(N_EPOCHS_TRAIN):

			log.start("Run %i"%run)


			# start a train epoch 
			DBM.train(	train_data  = train_data,
						train_label = train_label if LOAD_MNIST else None,
						num_batches = N_BATCHES_TRAIN,
						cont        = run)

			# test session while training
			if run!=N_EPOCHS_TRAIN-1 and run%TEST_EVERY_EPOCH==0:
				# wrong_classified_id = np.loadtxt("wrongs.txt").astype(np.int)
				# DBM.test(train_data[:1000], train_label[:1000], 50, 10)

				DBM.test(test_data, test_label if LOAD_MNIST else None,
						N = 10,  # sample ist aus random werten, also mindestens 2 sample machen 
						M = 10,  # average v
						create_conf_mat = 0)





			# DBM.test(test_data_noise) 

			log.end()

	DBM.train_time=log.end()
	log.reset()

# last test session
if DO_TESTING:
	with tf.Session() as sess:
		DBM.test(test_data, test_label if LOAD_MNIST else None,
				N = 40,  # sample ist aus random werten, also mindestens 2 sample machen 
				M = 20,  # average v. 0->1 sample
				create_conf_mat = 1)

if DO_GEN_IMAGES:
	with tf.Session() as sess:
		log.start("Generation Session")

		if DO_LOAD_FROM_FILE and not DO_TRAINING:
			DBM.load_from_file(workdir+"/data/"+PATHSUFFIX,override_params=1)		
		DBM.batchsize = 1
		DBM.graph_init("gibbs")
		DBM.import_()


		nn=5 ## grid with nn^2 plots
		fig,ax = plt.subplots(nn,nn)
		m=0
		for i in range(nn):
			for j in range(nn):
				generated_img = DBM.gibbs_sampling([[0,0,1.5,0,0,0,0,0,0,0]], 500, 0.1 , 0.1, 
							mode     = "freerunning",
							subspace = [],
							liveplot = 0)
				ax[i,j].matshow(generated_img.reshape(28, 28))
				ax[i,j].set_xticks([])
				ax[i,j].set_yticks([])
				ax[i,j].grid(False)
				m += 1
		# plt.tight_layout(pad=0.3)

		log.end()

if DO_CONTEXT:
	with tf.Session() as sess:
		log.start("Context Session")

		if DO_LOAD_FROM_FILE and not DO_TRAINING:
			DBM.load_from_file(workdir+"/data/"+PATHSUFFIX,override_params=1)

		subspace = [0,1,2,3,4]

		# p = 1
		# log.info("Multiplicator p = ",p)
		# context_mod = np.zeros(10)
		# for i in range(10):
		# 	if i in subspace:
		# 		context_mod[i] = 1*p

		# loop through images from all wrong classsified images and find al images that are <5 
		index_for_number_gibbs=[]
		for i in range(10000):
			## find the digit that was presented
			digit=np.where(test_label[i])[0][0] 		
			## set desired digit range
			if digit in subspace:
				index_for_number_gibbs.append(i)
		log.info("Found %i Images"%len(index_for_number_gibbs))


		# create graph 
		DBM.batchsize=len(index_for_number_gibbs)	
		if DBM.batchsize==0:
			raise ValueError("No Images found")

		DBM.graph_init("gibbs")
		DBM.import_()


		# calculte h2 firerates over all gibbs_steps 
		log.start("Sampling data")
		h2_no_context = DBM.gibbs_sampling(test_data[index_for_number_gibbs[:]], 1000, 0.05 , 0.05, 
							mode     = "context",
							subspace = "all",
							liveplot = 0)

		# # with context
		h2_context = DBM.gibbs_sampling(test_data[index_for_number_gibbs[:]], 1000, 0.05 , 0.05, 
							mode     = "context",
							subspace = subspace,
							liveplot = 0)
		# log.end()
		DBM.export()

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
			maxi_c    = h2_context[i][subspace[:]].max()
			max_pos_c = np.where(h2_context[i] == maxi_c)[0][0]
			if max_pos_c == digit:
				correct_maxis_c.append(maxi_c)
			else:
				if max_pos_c  not  in  subspace:
					wrongs_outside_subspace_c += 1
				incorrect_maxis_c.append(maxi_c)

			### count how many got right (no context) 
			## but only count the labels within subspace
			maxi_nc    = h2_no_context[i][subspace[:]].max()
			max_pos_nc = np.where(h2_no_context[i] == maxi_nc)[0][0]			
			if max_pos_nc == digit:
				correct_maxis_nc.append(maxi_nc)
			else:
				if max_pos_c  not in  subspace:
					wrongs_outside_subspace_nc += 1
				incorrect_maxis_nc.append(maxi_nc)

			desired_digits_c.append(h2_context[i,digit])
			desired_digits_nc.append(h2_no_context[i,digit])

			wrong_digits_c.append(np.mean(h2_context[i,digit+1:])+np.mean(h2_context[i,:digit]))
			wrong_digits_nc.append(np.mean(h2_no_context[i,digit+1:])+np.mean(h2_context[i,:digit]))

		log.info("Inorrect Context:" , len(incorrect_maxis_c),"/",round(100*len(incorrect_maxis_c)/float(len(index_for_number_gibbs)),2),"%")
		log.info("Inorrect No Context:" , len(incorrect_maxis_nc),"/",round(100*len(incorrect_maxis_nc)/float(len(index_for_number_gibbs)),2),"%")
		log.info("Diff:     ",len(incorrect_maxis_nc)-len(incorrect_maxis_c))
		log.info("Outside subspace (c/nc):",wrongs_outside_subspace_c,",", wrongs_outside_subspace_nc)
		log.out("Means: Correct // Wrong (c/nc): \n \t \t ", round(np.mean(correct_maxis_c),4),
									round(np.mean(correct_maxis_nc),4), "//",
									round(np.mean(incorrect_maxis_c),4),
									round(np.mean(incorrect_maxis_nc),4)
			)

		# calc how many digits got badly classified under a threshold 
		wrong_class_nc = [np.sum(np.array(desired_digits_nc)[:]<i) for i in np.linspace(0,1,1000)]
		wrong_class_c  = [np.sum(np.array(desired_digits_c)[:]<i)  for i in np.linspace(0,1,1000)]

		wrong_class_nc2 = [np.sum(np.array(wrong_digits_nc)[:]>i) for i in np.linspace(0,1,1000)]
		wrong_class_c2  = [np.sum(np.array(wrong_digits_c)[:]>i)  for i in np.linspace(0,1,1000)]


		plt.figure()
		plt.plot(np.linspace(0,1,1000),wrong_class_c,"-",label="With Context")
		plt.plot(np.linspace(0,1,1000),wrong_class_nc,"-",label="Without Context")
		plt.plot(np.linspace(0,1,1000),wrong_class_c2,"-",label="With Context / Mean")
		plt.plot(np.linspace(0,1,1000),wrong_class_nc2,"-",label="Without Context / Mean")
		plt.title("How many digits got classified below Threshold")
		plt.xlabel("Threshold")
		plt.ylabel("Number of Digits")
		plt.legend(loc="best")


		### plt histograms for each used digit
		fig,ax = plt.subplots(1,len(subspace),figsize=(12,7),sharey="row")
		plt.suptitle("With context")
		for i,digit in enumerate(subspace):
			ax[i].bar(range(10),np.mean(hist_data[digit][1:],axis=0))
			ax[i].set_ylim([0,0.3])
			ax[i].set_title(str(digit))
			ax[i].set_xticks(range(10))

		### plt histograms for each used digit
		fig,ax = plt.subplots(1,len(subspace),figsize=(12,7),sharey="row")
		plt.suptitle("Without context")
		for i,digit in enumerate(subspace):
			ax[i].bar(range(10),np.mean(hist_data_nc[digit][1:],axis=0))
			ax[i].set_ylim([0,0.3])
			ax[i].set_title(str(digit))
			ax[i].set_xticks(range(10))

		log.end()

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


if DO_TRAINING and DO_SAVE_TO_FILE:
	DBM.write_to_file()


####################################################################################################################################
#### Plot
# Plot the Weights, Errors and other informations
h1_shape = int(sqrt(DBM.SHAPE[1]))

log.out("Plotting...")

if DO_TRAINING:
	# plot w1 as image	
	fig=plt.figure(figsize=(9,9))
	map1=plt.matshow(tile(DBM.w_np[0]),cmap="gray",fignum=fig.number)
	plt.colorbar(map1)
	plt.grid(False)
	plt.title("W %i"%0)
	save_fig(saveto_path+"/weights_img.pdf", DO_SAVE_TO_FILE)

	# plot layer diversity
	plt.figure("Layer diversity")
	for i in range(DBM.n_layers):
		plt.plot(smooth(np.array(DBM.layer_diversity_)[::2,i],10),label="Layer %i"%i)
		plt.legend()
	plt.xlabel("Update Number")
	plt.ylabel("Deviation")
	save_fig(saveto_path+"/layer_diversity.pdf", DO_SAVE_TO_FILE)	

	plt.figure("Errors")
	## train errors
	plt.plot(DBM.recon_error_train[:],"-",label="Recon Error Train")
	if DBM.classification:
		plt.plot(DBM.class_error_train[:],"-",label="Class Error Train")
	## test errors
	# calc number of updates per epoch
	n_u_p_e = len(DBM.recon_error_train) / DBM.epochs
	x = np.array(DBM.save_dict["Test_Epoch"])*n_u_p_e
	plt.plot(x,DBM.save_dict["Recon_Error"],"o--",label="Recon Error Test")
	if DBM.classification:
		plt.plot(x,DBM.save_dict["Class_Error"],"o--",label="Class Error Test")
	plt.legend(loc="best")
	plt.xlabel("Update Number")
	plt.ylabel("Mean Square Error")
	save_fig(saveto_path+"/errors.pdf", DO_SAVE_TO_FILE)


	# plot all other weights as hists
	n_weights=DBM.n_layers-1
	fig,ax = plt.subplots(n_weights,1,figsize=(8,10),sharex="col")
	for i in range(n_weights):
		if n_weights>1:
			ax[i].hist((DBM.w_np[i]).flatten(),bins=60,alpha=0.5,label="After Training")
			ax[i].set_title("W %i"%i)
			ax[i].legend()
		else:
			ax.hist((DBM.w_np[i]).flatten(),bins=60,alpha=0.5,label="After Training")
			ax.set_title("W %i"%i)
			ax.legend()

		try:
			ax[i].hist((DBM.w_np_old[i]).flatten(),color="r",bins=60,alpha=0.5,label="Before Training")
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



	fig,ax = plt.subplots(3,1,sharex="col")

	ax[0].plot(DBM.save_dict["Temperature"],label="Temperature")
	ax[0].legend(loc="center left",bbox_to_anchor = (1.0,0.5))

	ax[0].set_ylabel("Temperature")

	ax[1].plot(DBM.save_dict["Learnrate"],label="Learnrate")
	ax[1].legend(loc="center left",bbox_to_anchor = (1.0,0.5))
	ax[1].set_ylabel("Learnrate")

	ax[2].set_ylabel("Weights Mean")
	for i in range(len(DBM.SHAPE)-1):
		ax[2].plot(DBM.save_dict["W_mean_%i"%i],label="Weight %i"%i)
	ax[2].legend(loc="center left",bbox_to_anchor = (1.0,0.5))
	ax[2].set_xlabel("Epoch")
	plt.subplots_adjust(bottom=None, right=0.73, top=None,
	            wspace=None, hspace=None)
	save_fig(saveto_path+"/learnr-temp.pdf", DO_SAVE_TO_FILE)


	plt.figure("Layer_activiations_test_run")
	for i in range(DBM.n_layers):
		plt.plot(DBM.layer_act[:,i],label="Layer %i"%i)
	plt.legend()


#plot only one digit
if LOAD_MNIST:
	#plot some samples from the testdata 
	fig3,ax3 = plt.subplots(len(DBM.SHAPE)+1,13,figsize=(16,4),sharey="row")
	for i in range(13):
		# plot the input
		ax3[0][i].matshow(test_data[i:i+1].reshape(int(sqrt(DBM.SHAPE[0])),int(sqrt(DBM.SHAPE[0]))))
		ax3[0][i].set_yticks([])
		ax3[0][i].set_xticks([])
		# plot the reconstructed image		
		ax3[1][i].matshow(DBM.probs[i:i+1].reshape(int(sqrt(DBM.SHAPE[0])),int(sqrt(DBM.SHAPE[0]))))
		ax3[1][i].set_yticks([])
		ax3[1][i].set_xticks([])
		
		#plot all layers that can get imaged
		for layer in range(len(DBM.SHAPE)-1):
			try:
				ax3[layer+2][i].matshow(DBM.hidden_save[layer][i:i+1].reshape(int(sqrt(DBM.SHAPE[layer+1])),int(sqrt(DBM.SHAPE[layer+1]))))
				ax3[layer+2][i].set_yticks([])
				ax3[layer+2][i].set_xticks([])
			except:
				pass
		# plot the last layer 	
		if DBM.classification:	
			ax3[-1][i].bar(range(DBM.SHAPE[-1]),DBM.last_layer_save[i])
			ax3[-1][i].set_xticks(range(DBM.SHAPE[-1]))
			ax3[-1][i].set_ylim(0,1)
		else:
			ax3[-1][i].matshow(DBM.last_layer_save[i].reshape(int(sqrt(DBM.SHAPE[-1])),int(sqrt(DBM.SHAPE[-1]))))
			ax3[-1][i].set_xticks([])
			ax3[-1][i].set_yticks([])

		#plot the reconstructed layer h1
		# ax3[5][i].matshow(DBM.rec_h1[i:i+1].reshape(int(sqrt(DBM.SHAPE[1])),int(sqrt(DBM.SHAPE[1]))))
		# plt.matshow(random_recon.reshape(int(sqrt(DBM.SHAPE[0])),int(sqrt(DBM.SHAPE[0]))))
	plt.tight_layout(pad=0.0)
	save_fig(saveto_path+"/examples.pdf", DO_SAVE_TO_FILE)


	fig3,ax3 = plt.subplots(len(DBM.SHAPE)+1,10,figsize=(16,4),sharey="row")
	m=0
	for i in index_for_number_test.astype(np.int)[8][0:10]:
		# plot the input
		ax3[0][m].matshow(test_data[i:i+1].reshape(int(sqrt(DBM.SHAPE[0])),int(sqrt(DBM.SHAPE[0]))))
		ax3[0][m].set_yticks([])
		ax3[0][m].set_xticks([])
		# plot the reconstructed image		
		ax3[1][m].matshow(DBM.probs[i:i+1].reshape(int(sqrt(DBM.SHAPE[0])),int(sqrt(DBM.SHAPE[0]))))
		ax3[1][m].set_yticks([])
		ax3[1][m].set_xticks([])
		
		#plot all layers that can get imaged
		for layer in range(len(DBM.SHAPE)-1):
			try:
				ax3[layer+2][m].matshow(DBM.hidden_save[layer][i:i+1].reshape(int(sqrt(DBM.SHAPE[layer+1])),int(sqrt(DBM.SHAPE[layer+1]))))
				ax3[layer+2][m].set_yticks([])
				ax3[layer+2][m].set_xticks([])
			except:
				pass
		# plot the last layer 		
		if DBM.classification:
			ax3[-1][m].bar(range(DBM.SHAPE[-1]),DBM.last_layer_save[i])
			ax3[-1][m].set_xticks(range(DBM.SHAPE[-1]))
			ax3[-1][m].set_ylim(0,1)
		#plot the reconstructed layer h1
		# ax4[5][m].matshow(DBM.rec_h1[i:i+1].reshape(int(sqrt(DBM.SHAPE[1])),int(sqrt(DBM.SHAPE[1]))))
		# plt.matshow(random_recon.reshape(int(sqrt(DBM.SHAPE[0])),int(sqrt(DBM.SHAPE[0]))))
		m+=1
	plt.tight_layout(pad=0.0)





log.close()
if DO_SHOW_PLOTS:
	plt.show()
else:
	plt.close()
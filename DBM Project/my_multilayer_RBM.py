# -*- coding: utf-8 -*-
#### Imports
if True:
	# -*- coding: utf-8 -*-
	print ("Starting...")
	import numpy as np
	import numpy.random as rnd
	import matplotlib.pyplot as plt
	import matplotlib as mpl
	import tensorflow as tf
	# import scipy.ndimage.filters as filters
	# import pandas as pd
	import os,time,sys
	from math import exp,sqrt,sin,pi,cos,log
	np.set_printoptions(precision=3)
	
	workdir="/Users/Niermann/Google Drive/Masterarbeit/Python/DBM Project"
	# workdir="/home/dario/Dokumente/DBM Project"

	data_dir=workdir+"/data"	
	os.chdir(workdir)
	from Logger import *
	from RBM_Functions import *
	### import seaborn? ###
	if 1:
		import seaborn

		seaborn.set(font_scale=1.3)
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
if "train_data" not in globals():
	log.out("Loading Data")
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	train_data, train_label, test_data, test_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
	#get test data of only one number class:
	index_for_number_test  = []
	index_for_number_train = []
	for i in range(len(test_label)):
		if (test_label[i]==[0,0,0,1,0,0,0,0,0,0]).sum()==10:
			index_for_number_test.append(i)
	for i in range(len(train_label)):
		if (train_label[i]==[0,0,0,1,0,0,0,0,0,0]).sum()==10:
			index_for_number_train.append(i)


	test_data_noise = np.copy(test_data)
	# making noise 
	for i in range(len(test_data_noise)):
		test_data_noise[i]  += np.round(rnd.random(test_data_noise[i,:].shape)*0.55)
		# half_images[i] = abs(half_images[i])
		# half_images[i] *= 1/half_images[i].max()
		# half_images[i] *= rnd.random(half_images[i].shape)
	test_data_noise   = test_data_noise>0

	noise_data_train = sample_np(rnd.random(train_data.shape)*0.2)
	noise_data_test = sample_np(rnd.random(test_data.shape)*0.2)
	noise_label_train = np.zeros(train_label.shape)
	noise_label_test = np.zeros(test_label.shape)

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

		self.v       = tf.placeholder(tf.float32,[None,self.visible_units],name="Visible-Layer") # has shape [number of images per batch,number of visible units]

		self.w       = tf.Variable(tf.random_uniform([self.visible_units,self.hidden_units],minval=-1e-3,maxval=1e-3),name="Weights")
		self.bias_v  = tf.Variable(tf.zeros([self.visible_units]),name="Visible-Bias")
		self.bias_h  = tf.Variable(tf.zeros([self.hidden_units]), name="Hidden-Bias")


		# get the probabilities of the hidden units in 
		self.h_prob  = sigmoid(tf.matmul(self.v,self.forw_mult*self.w) + self.bias_h,temp)
		# h has shape [number of images per batch, number of hidden units]
		# get the actual activations for h {0,1}
		self.h       = tf.nn.relu(
			            tf.sign(
			            	self.h_prob - tf.random_uniform(tf.shape(self.h_prob)) 
			            	) 
		        		) 

		# and the same for visible units
		self.v_prob  = sigmoid(tf.matmul(self.h,(self.back_mult*self.w),transpose_b=True) + self.bias_v,temp)
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
		self.pos_grad  = tf.matmul(self.v,self.h,transpose_a=True)
		self.neg_grad  = tf.matmul(self.v_recon,self.h_gibbs,transpose_a=True)
		self.numpoints = tf.cast(tf.shape(self.v)[0],tf.float32) #number of train inputs per batch (for averaging the CD matrix -> see practical paper by hinton)
		# contrastive divergence
		self.CD        = (self.pos_grad - self.neg_grad)/self.numpoints
		

		#update w
		self.update_w = self.w.assign(self.w+self.learnrate*self.CD)
		self.mean_w   = tf.reduce_mean(self.w)

		#update bias
		""" Since vectors v and h are actualy matrices with number of batch_size images in them, reduce mean will make them to a vector again """
		# self.update_bias_v = self.bias_v.assign(self.bias_v+self.learnrate*tf.reduce_mean(self.v-self.v_recon,0))
		# self.update_bias_h = self.bias_h.assign(self.bias_h+self.learnrate*tf.reduce_mean(self.h-self.h_gibbs,0))


		# reverse feed
		# self.h_rev       = tf.placeholder(tf.float32,[None,self.hidden_units],name="Reverse-hidden")
		# self.v_prob_rev  = sigmoid(tf.matmul(self.h_rev,(self.w),transpose_b=True) + self.bias_v,temp)
		# self.v_recon_rev = tf.nn.relu(tf.sign(self.v_prob_rev - tf.random_uniform(tf.shape(self.v_prob_rev))))

	def train(self,sess,RBM_i,RBMs,batch):
		self.my_input_data = batch
		# iterate which RBM level this is and calculate the proper input 
		for j in range(1,len(RBMs)):
			if RBM_i >= j:
				self.my_input_data = RBMs[j-1].h.eval({RBMs[j-1].v : self.my_input_data})

		#### update the weights and biases
		self.w_i, self.error_i = sess.run([self.update_w,self.error],feed_dict={self.v:self.my_input_data})
		# sess.run([self.update_bias_h,self.update_bias_v],feed_dict={self.v:self.my_input_data})

		return self.w_i,self.error_i




################################################################################################################################################
### Class Deep BM 
class DBM_class(object):
	"""defines a deep boltzmann machine
	"""

	def __init__(self,shape,liveplot):
		self.n_layers     = len(shape)
		self.liveplot     = liveplot # if true will open a lifeplot of the weight matrix 
		self.shape        = shape  # contains the number of  neurons in a list from v layer to h1 to h2 
		
		
		
		self.init_state     = 0
		self.exported       = 0
		self.m              = 0 #laufvariable
		self.num_of_skipped = 10 # how many tf.array value adds get skipped 
		self.train_time     = 0
		self.test_error_ = []

		self.log_list = [	["n_units_first_layer",shape[0]],
					["n_units_second_layer",shape[1]],
					["n_units_third_layer",shape[2]],
					["epochs_pretrain",pretrain_epochs],
					["epochs_dbm_train",dbm_epochs],
					["batches_pretrain",num_batches_pretrain],
					["batches_dbm_train",dbm_batches],
					["learnrate_pretrain",rbm_learnrate],
					["learnrate_dbm_train",dbm_learnrate],
					["learnrate_dbm_train_end",dbm_learnrate_end],
					["Temperature",temp],
					["pathsuffix_pretrained",pathsuffix_pretrained],
					["pathsuffix",pathsuffix],
					["loaded_from_file",load_from_file],
					["save_all_params",save_all_params]
				   ]

		self.RBMs    = [None]*(len(self.shape)-1)
		for i in range(len(self.RBMs)):
			self.RBMs[i] = RBM(self.shape[i],self.shape[i+1], forw_mult= 1, back_mult = 1, learnrate = rbm_learnrate, liveplot=0)
		# self.RBMs[1] = RBM(self.shape[1],self.shape[2], forw_mult= 1, back_mult = 1, learnrate = rbm_learnrate, liveplot=0)
		# self.RBMs[2] = RBM(self.shape[2],self.shape[3], forw_mult= 1, back_mult = 1, learnrate = rbm_learnrate, liveplot=0)

		log.out("not using forw and backw multiplicators")

	def pretrain(self):
		""" this function will pretrain the RBMs and define a self.weights list where every
		weight will be stored in. This weights list can then be used to save to file and/or 
		to be loaded into the DBM for further training. 
		"""

		if pre_training:
			for rbm in self.RBMs:
				if rbm.liveplot:
					log.info("Liveplot is open!")
					fig,ax=plt.subplots(1,1,figsize=(15,10))
					break

		batchsize_pretrain = int(55000/num_batches_pretrain)

		with tf.Session() as sess:
			# train session - v has batchsize length
			log.start("Pretrain Session")
			
			
			#iterate through the RBMs , each iteration is a RBM
			if pre_training:	
				sess.run(tf.global_variables_initializer())

				for RBM_i, RBM in enumerate(self.RBMs):
					log.start("Pretraining ",str(RBM_i+1)+".", "RBM")
					

					for epoch in range(pretrain_epochs[RBM_i]):

						log.start("Epoch:",epoch+1,"/",pretrain_epochs[RBM_i])
						
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


						log.info("Learnrate:",round(rbm_learnrate,4))
						log.info("error",round(error_i,4))
						log.end() #ending the epoch


					log.end() #ending training the rbm 

				

				# define the weights
				self.weights  =  []
				for i in range(len(self.RBMs)):
					self.weights.append(self.RBMs[i].w.eval())

				if save_pretrained:
					for i in range(len(self.weights)):
						np.savetxt(workdir+"/pretrain_data/"+"Pretrained-"+" %i "%i+str(time_now)+".txt", self.weights[i])
					log.out("Saved Pretrained under "+str(time_now))
			else:
				if not load_from_file:
					### load the pretrained weights
					self.weights=[]
					log.out("Loading Pretrained from file")
					for i in range(len(self.shape)-1):
						self.weights.append(np.loadtxt(workdir+"/pretrain_data/"+"Pretrained-"+" %i "%i+pathsuffix_pretrained+".txt").astype(np.float32))
				else:
					### if loading from file is active the pretrained weights would get 
					### reloaded anyway so directly load them here
					self.weights=[]
					log.out("Loading from file")
					for i in range(len(self.shape)-1):
						self.weights.append(np.loadtxt(data_dir+"/"+pathsuffix+"/"+"w%i.txt"%(i)).astype(np.float32))
			log.end()


	def load_from_file(self,path):
		os.chdir(path)
		log.out("Loading data from:","...",path[-20:])

		self.w_np     = []
		self.w_np_old = []
		for i in range(len(self.shape)-1):
			self.w_np.append(np.loadtxt("w%i.txt"%(i)))
			self.w_np_old.append(self.w_np[i])  #save weights for later comparison

		self.bias_np = []
		for i in range(len(self.shape)):
			self.bias_np.append(np.loadtxt("bias%i.txt"%(i)))

		os.chdir(workdir)

	
	def import_(self):
		""" setting up the graph and setting the weights and biases tf variables to the 
		saved numpy arrays """
		log.out("loading numpy vars into graph")
		for i in range(len(self.shape)-1):
			sess.run(self.w[i].assign(self.w_np[i]))
		for i in range(len(self.shape)):
			sess.run(self.bias[i].assign(self.bias_np[i]))

	def layer_input(self, layer_i):
		""" calculate input of layer layer_i
		layer_i :: for which layer
		returns :: input for the layer - which are the probabilites
		"""
		if layer_i == 0:
			_input_ = sigmoid(tf.matmul(self.layer[layer_i+1], self.w[layer_i],transpose_b=True) + self.bias[layer_i], self.temp)			

		elif layer_i == self.n_layers-1:
			_input_ = sigmoid(tf.matmul(self.layer[layer_i-1],self.w[layer_i-1]) + self.bias[layer_i], self.temp)
		
		else:
			_input_ = sigmoid(tf.matmul(self.layer[layer_i-1],self.w[layer_i-1]) 
					+ tf.matmul(self.layer[layer_i+1],self.w[layer_i],transpose_b=True) 
					+ self.bias[layer_i], 
					self.temp)
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

	################################################################################################################################################
	####  DBM Graph 
	################################################################################################################################################
	def graph_init(self,graph_mode):
		""" sets the graph up and loads the pretrained weights in , these are given
		at class definition
		graph_mode  :: "training" if the graph is used in training - this will set h2 to placeholder for the label data
				:: "testing"  if the graph is used in testing - this will set h2 to a random value and to be calculated from h1 
				:: "gibbs"    if the graph is used in gibbs sampling - this will set temperature to a placeholder
		"""
		log.out("Initializing graph")
		

		# self.v  = tf.placeholder(tf.float32,[self.batchsize,self.shape[0]],name="Visible-Layer") 

		if graph_mode=="training":
			# stuff
			self.m_tf      = tf.placeholder(tf.int32,[],name="running_array_index")
			self.learnrate = tf.placeholder(tf.float32,[],name="Learnrate")

			# arrays for saving progress
			self.h1_activity_ = tf.Variable(tf.zeros([self.num_of_updates/self.num_of_skipped]))
			self.h2_activity_ = tf.Variable(tf.zeros([self.num_of_updates/self.num_of_skipped]))
			self.train_error_ = tf.Variable(tf.zeros([self.num_of_updates/self.num_of_skipped]))
			self.train_class_error_ = tf.Variable(tf.zeros([self.num_of_updates/self.num_of_skipped]))
			
		#### temperature
		if graph_mode=="gibbs" or graph_mode=="testing":
			self.temp = tf.placeholder(tf.float32,[],name="Temperature")
		else:
			self.temp = temp



		### init all Parameters like weights , biases , layers and their updates
		# weights
		self.w               = [None]*(self.n_layers-1)
		self.pos_grad        = [None]*(self.n_layers-1)
		self.neg_grad        = [None]*(self.n_layers-1)
		self.update_pos_grad = [None]*(self.n_layers-1)
		self.update_neg_grad = [None]*(self.n_layers-1)
		self.update_w        = [None]*(self.n_layers-1)
		self.w_mean_ 	   = [None]*(self.n_layers-1) # variable to store means
		self.mean_w          = [None]*(self.n_layers-1) # calc of mean for each w

		# bias
		self.bias        = [None]*self.n_layers
		self.update_bias = [None]*self.n_layers

		# layer
		self.layer      = [None]*self.n_layers # layer variable 
		self.layer_save = [None]*self.n_layers # save variable (used for storing older layers)
		self.save_layer = [None]*self.n_layers # save variable (used for storing older layers)
		self.layer_ph   = [None]*self.n_layers # placeholder 
		self.assign_l   = [None]*self.n_layers # assign op. (assigns placeholder)
		self.layer_prob = [None]*self.n_layers # calc prob for layer n
		self.layer_samp = [None]*self.n_layers # take a sample from the prob
		self.update_l_s = [None]*self.n_layers # assign op. for calculated samples
		self.update_l_p = [None]*self.n_layers # assign op. for calculated probs
		self.layer_activities = [None]*self.n_layers # calc for layer activieties (mean over batch)
		self.layer_energy = [None]*(self.n_layers-1)

		### layer vars 
		for i in range(len(self.layer)):
			self.layer[i]      = tf.Variable(tf.random_uniform([self.batchsize,self.shape[i]],minval=-1e-3,maxval=1e-3),name="Layer_%i"%i)
			self.layer_save[i] = tf.Variable(tf.random_uniform([self.batchsize,self.shape[i]],minval=-1e-3,maxval=1e-3),name="Layer_save_%i"%i)
			self.layer_ph[i]   = tf.placeholder(tf.float32,[self.batchsize,self.shape[i]],name="layer_%i_PH"%i)

		### weight calculations and assignments
		for i in range(len(self.w)):
			self.w[i] = tf.Variable(self.weights[i],name="Weights%i"%i)
			if graph_mode=="training":
				self.pos_grad[i] = tf.Variable(tf.zeros([self.shape[i],self.shape[i+1]]))
				self.neg_grad[i] = tf.Variable(tf.zeros([self.shape[i],self.shape[i+1]]))
				self.update_pos_grad[i] = self.pos_grad[i].assign(tf.matmul(self.layer[i], self.layer[i+1],transpose_a=True))
				self.update_neg_grad[i] = self.neg_grad[i].assign(tf.matmul(self.layer[i], self.layer[i+1],transpose_a=True))
				self.update_w[i] = self.w[i].assign_add(self.learnrate*(self.pos_grad[i] - self.neg_grad[i])/self.batchsize)
				self.w_mean_[i]   = tf.Variable(tf.zeros([self.num_of_updates/self.num_of_skipped]))
				self.mean_w[i]   = tf.reduce_mean(tf.square(self.w[i]))

		### bias calculations and assignments
		for i in range(len(self.bias)):
			self.bias[i] = tf.Variable(tf.zeros([self.shape[i]]),name="Bias%i"%i)
			if graph_mode == "training":
				self.update_bias[i] = self.bias[i].assign_add(self.learnrate*tf.reduce_mean(tf.subtract(self.layer_save[i],self.layer[i]),0))

		### layer calculations and assignments
		for i in range(len(self.layer)):
			self.save_layer[i] = self.layer_save[i].assign(self.layer[i])
			self.assign_l[i]   = self.layer[i].assign(self.layer_ph[i])
			self.layer_prob[i] = self.layer_input(i)
			self.layer_samp[i] = self.sample(self.layer_prob[i])
			self.update_l_p[i] = self.layer[i].assign(self.layer_prob[i])
			self.layer_activities[i] = tf.reduce_mean(tf.reduce_sum(self.layer[i],1)/self.shape[i])
		for i in range(len(self.layer)-1):
			self.layer_energy[i] = tf.matmul(self.layer[i], tf.matmul(self.w[i],self.layer[i+1],transpose_b=True))
			self.update_l_s[i]   = self.layer[i].assign(self.layer_samp[i])
		self.update_l_s[-1] = self.layer[-1].assign(self.layer_prob[-1])

		# modification array size 10 that gehts multiplied to the label vector for context
		self.modification_tf = tf.Variable(tf.ones([self.batchsize,self.shape[-1]]),name="Modification")



		### Error and stuff
		self.error       = tf.reduce_mean(tf.square(self.layer_ph[0]-self.layer[0]))
		self.class_error = tf.reduce_mean(tf.square(self.layer_ph[-1]-self.layer[-1]))

		self.h1_sum    = tf.reduce_sum(self.layer[1])
		self.h2_sum    = tf.reduce_sum(self.layer[2])
		self.label_sum = tf.reduce_sum(self.layer[-1])

		self.free_energy = -tf.reduce_sum(tf.log(1+tf.exp(tf.matmul(self.layer[0],self.w[0])+self.bias[1])))

		self.energy = -tf.reduce_sum([self.layer_energy[i] for i in range(len(self.layer_energy))])

		#### updates for each layer 
		self.update_h2_with_context = self.layer[-2].assign(self.sample(sigmoid(tf.matmul(self.layer[-3],self.w[-2])  
											+ tf.matmul(tf.multiply(self.layer[-1],self.modification_tf),self.w[-1],transpose_b=True)
											+ self.bias[-2],self.temp)))
	

		### Training with contrastive Divergence
		if graph_mode=="training":
			self.assign_arrays =	[ tf.scatter_update(self.train_error_,self.m_tf,self.error),
							  tf.scatter_update(self.train_class_error_,self.m_tf,self.class_error),							  
							  tf.scatter_update(self.h1_activity_,self.m_tf,self.h1_sum),
							]
			for i in range(len(self.shape)-1):
				self.assign_arrays.append(tf.scatter_update(self.w_mean_[i],self.m_tf,self.mean_w[i]))
		

		sess.run(tf.global_variables_initializer())
		self.init_state=1

	def test_noise_stability(self,input_data,input_label):
		self.batchsize=len(input_data)
		if load_from_file:
			self.load_from_file(workdir+"/data/"+pathsuffix)
		self.graph_init("testing")
		self.import_()

		n       = 20
		h2_     = []
		r       = rnd.random([self.batchsize,784])
		v_noise = np.copy(input_data)
		
		for i in range(200):
			h2            = self.h2_prob.eval({self.v:v_noise})
			v_noise_recon = self.v_recon_prob.eval({self.v:v_noise})
			
			for i in range(n):
				v_noise_recon+=self.v_recon_prob.eval({self.v:v_noise})
			v_noise_recon*=1./(n+1)
			
			# classify the reconstructed image
			for i in range(n):	
				h2 += self.h2_prob.eval({self.v:v_noise_recon})
			h2*=1./(n+1)
			
			
			# make the input more noisy
			v_noise += (abs(r-0.5)*0.01)
			v_noise *= 1./v_noise.max()

			h2_.append(h2[0])
		
		return np.array(h2_),v_noise_recon


	def train(self,train_data,train_label,epochs,num_batches,learnrate,N,cont):
		""" training the DBM with given h2 as labels and v as input images
		train_data  :: images
		train_label :: corresponding label
		epochs      :: how many epochs to train
		num_batches :: how many batches
		learnrate   :: learnrate
		N           :: Number of gibbs steps
		"""
		######## init all vars for training
		self.batchsize = int(55000/num_batches)
		num_of_updates = epochs*num_batches
		M              = 20

		log.info("Batchsize:",self.batchsize,"NBatches",num_of_updates)

		self.num_of_updates = num_of_updates
		d_learnrate         = float(dbm_learnrate_end-learnrate)/num_of_updates
		self.m              = 0

		## arrays for approximation of hidde probs
		h1_p=np.zeros([M,self.batchsize,self.shape[1]])
		h2_p=np.zeros([M,self.batchsize,self.shape[2]])


		### free energy
		# self.F=[]
		# self.F_test=[]
		
		if load_from_file:
			# load data from the file
			self.load_from_file(workdir+"/data/"+pathsuffix)
			self.graph_init("training")
			self.import_()

		# if no files loaded then init the graph with pretrained vars
		if self.init_state==0:
			self.graph_init("training")

		if cont:
			self.graph_init("training")
			self.import_()


		if self.liveplot:
			log.info("Liveplot is on!")
			fig,ax = plt.subplots(1,1,figsize=(15,10))
			data   = ax.matshow(tile(self.w1.eval()), vmin=-0.01, vmax=0.01)
			plt.colorbar(data)


		# starting the training
		for epoch in range(epochs):
			log.start("Deep BM Epoch:",epoch+1,"/",epochs)

			# shuffle test data and labels so that batches are not equal every epoch 
			log.out("Shuffling TrainData")
			self.seed   = rnd.randint(len(train_data),size=(int(len(train_data)/10),2))
			train_data  = shuffle(train_data, self.seed)
			train_label = shuffle(train_label, self.seed)

			log.out("Running Batch")
			# log.info("++ Using Weight Decay! Not updating bias! ++")
			log.info("Freerunning for %i steps"%N)

				
			for start, end in zip( range(0, len(train_data), self.batchsize), range(self.batchsize, len(train_data), self.batchsize)):
				# define a batch
				batch = train_data[start:end]
				batch_label = train_label[start:end]

				# assign v and h2 to the batch data
				sess.run([self.assign_l[0],self.assign_l[-1]],{ self.layer_ph[0]  : batch, 
										 	      self.layer_ph[-1] : batch_label})

				# calc hidden layer probabilities (not the visible & label layer)
				for hidden in range(M):
					sess.run(self.update_l_p[1:-1])
					# if hidden>0:
					# 	diff_h2 = np.mean(h2_p[hidden])-np.mean(h2_p[hidden-1])

				# save all layer for bias update
				sess.run(self.save_layer)

				# update the positive gradients
				sess.run(self.update_pos_grad)


				# update all layers N times (free running, gibbs sampling) 
				for n in range(N):
					# using sampling
					sess.run(self.update_l_s)
				# las step to reduce some sampling noise update h1 but only calc probs
				# sess.run([self.update_h1_probs,self.update_h2_probs])


				# calc he negatie gradients
				sess.run(self.update_neg_grad)


				# run all parameter updates 
				sess.run([self.update_w, self.update_bias],
						feed_dict={	self.layer_ph[0] : batch,
								self.layer_ph[-1] : batch_label,
								self.learnrate : learnrate}
					)


				#### calculate free energy for test and train data
				# self.F.append(self.free_energy.eval({self.v:batch}))
				# f_test_ = self.free_energy.eval({self.v:test_data[0:self.batchsize]})
				# for i in range(1,10):
				# 	f_test_ += self.free_energy.eval({self.v:test_data[i*self.batchsize:i*self.batchsize+self.batchsize]})
				# f_test_*=1./10
				# self.F_test.append(f_test_)


				# #### add values to the tf.arrays
				if self.m%self.num_of_skipped==0:
					try:
						sess.run([self.assign_arrays],
								feed_dict={	self.layer_ph[0] :      batch,
										self.layer_ph[-1]: batch_label,
										self.m_tf:           self.m / self.num_of_skipped }
							)
					except:
						log.info("Error for m="+str(self.m))

				# increase the learnrate
				learnrate += d_learnrate
				# increase index for tf arrays
				self.m += 1

				### liveplot
				if self.liveplot and plt.fignum_exists(fig.number) and start%40==0:
					if start%4000==0:
						ax.cla()
						data = ax.matshow(tile(self.w1.eval()),vmin=tile(self.w1.eval()).min()*1.2,vmax=tile(self.w1.eval()).max()*1.2)

					matrix_new = tile(self.w1.eval())
					data.set_data(matrix_new)
					plt.pause(0.00001)


			# self.train_error_np=self.train_error_.eval()
			# log.out("error:",np.round(self.train_error_np[m],4)," learnrate:",self.learnrate)
			log.info("Learnrate: ",learnrate)
			log.end() #ending the epoch

		log.reset()

		# normalize the activity arrays
		self.h1_activity_*=1./(self.shape[1]*self.batchsize)

		self.export()


	def test(self,test_data,test_label,N,M):
		""" testing runs without giving h2 , only v is given and h2 has to be infered 
		by the DBM 
		test_data :: images to test, get assigned to v layer
		N :: Number of updates from hidden layers 
		M :: Number of samples taken to reconstruct the image
		"""
		#init the vars and reset the weights and biases 		
		self.batchsize=len(test_data)
		self.learnrate = dbm_learnrate

		# h1    = np.zeros([N,self.batchsize,self.shape[1]])
		# h2    = np.zeros([N,self.batchsize,self.shape[2]])
		# label = np.zeros([N,self.batchsize,self.shape[-1]])

		self.label_diff = np.zeros([N,self.batchsize,self.shape[-1]])


		### init the graph 
		if load_from_file and not training:
			self.load_from_file(workdir+"/data/"+pathsuffix)
		self.graph_init("testing") # "testing" because this graph creates the testing variables where only v is given, not h2
		self.import_()


		#### start test run
		log.start("Testing DBM with %i images"%self.batchsize)
		
		#### give input to v layer
		sess.run(self.assign_l[0], {self.layer_ph[0] : test_data, self.temp : temp})

		#### update hidden and label N times
		log.out("Sampling hidden %i times "%N)
		self.layer_act = np.zeros([N,self.n_layers])
		self.save_h1 = []
		
		for n in range(N):
			self.hidden_save = sess.run([self.update_l_p[i] for i in range(1,len(self.shape))], {self.temp : temp})
			self.layer_act[n,:] = sess.run(self.layer_activities,{self.temp : temp})
			self.save_h1.append(self.layer[1].eval()[0])
		

		
		self.h1_test = self.hidden_save[0]
		self.label_test = self.hidden_save[-1]

		#### update v M times
		self.probs = self.layer[0].eval()
		self.image_timeline = []
		for i in range(M):
			self.probs += sess.run(self.update_l_s[0],{self.temp : temp})
			self.image_timeline.append(self.layer[0].eval()[0])
		self.probs *= 1./(M+1)



		#### calculate errors and activations
		self.test_error  = self.error.eval({self.layer_ph[0] : test_data})
		self.test_error_.append(self.test_error) #append to errors if called multiple times
		# error of classifivation labels
		self.class_error=np.mean(np.abs(self.label_test-test_label))		

		#### count how many images got classified wrong 
		log.out("Taking only the maximum")
		n_wrongs=0
		# label_copy=np.copy(self.label_test)
		wrong_classified_ind=[]
		wrong_maxis=[]

		for i in range(len(self.label_test)):
			digit = np.where(test_label[i]==1)[0][0]
			maxi    = self.label_test[i].max()
			max_pos = np.where(self.label_test[i] == maxi)[0][0]
			if max_pos != digit:
				wrong_classified_ind.append(i)
				wrong_maxis.append(maxi)
		n_wrongs=len(wrong_maxis)


		log.end()
		log.reset()
		log.info("Reconstr. error: ",np.round(self.test_error,5), "learnrate: ",np.round(dbm_learnrate,5))
		log.info("Class error: ",np.round(self.class_error,5))
		log.info("Wrong Digits: ",n_wrongs," with average: ",round(np.mean(wrong_maxis),3))
		# log.info("Activations of Neurons: ", np.round(self.h1_act_test,4) , np.round(self.label_act_test,4))
		return wrong_classified_ind


	def gibbs_sampling(self,v_input,gibbs_steps,temp_start,temp_end,modification,mode,liveplot=1):
		""" Repeatedly samples v and label , where label can be modified by the user with the multiplication
		by the modification array - clamping the labels to certain numbers.
		v_input :: starting with an image as input can also be a batch of images
		
		temp_end, temp_start :: temperature will decrease or increase to temp_end and start at temp_start 
		
		mode 	:: "sampling" calculates h2 and v back and forth usign previous steps
			:: "context" clamps v and only calculates h1 based on previous h2
		
		p :: multiplication of the h2 array to increase the importance of the layer
		"""

		self.layer_save = []
		for i in range(len(self.shape)):
			self.layer_save.append(np.zeros([gibbs_steps,self.batchsize,self.shape[i]]))

		temp_          = np.zeros([gibbs_steps])
		self.energy_   = []
		self.mean_h1   = []
		temp           = temp_start
		temp_delta     = (temp_end-temp_start)/gibbs_steps

		self.num_of_updates = 1000 #just needs to be defined because it will make a train graph with tf.arrays where this number is needed


		if liveplot:
			log.info("Liveplotting gibbs sampling")
			fig,ax=plt.subplots(1,len(self.shape)+1,figsize=(15,5))
			plt.tight_layout()



		if mode=="context":
			sess.run(self.assign_l[0],{self.layer_ph[0] : v_input})
			for i in range(1,len(self.shape)):
				sess.run( self.assign_l[i], {self.layer_ph[i] : 0.01*rnd.random([self.batchsize, self.shape[i]])} )
			
			modification = np.concatenate((modification,)*self.batchsize).reshape(self.batchsize,10)
			sess.run(self.modification_tf.assign(modification))

			for step in range(gibbs_steps):
				# update all layer except the last one 
				layer_1 = sess.run(self.update_l_s[1:-2], {self.temp : temp})
				layer_2 = sess.run(self.update_h2_with_context,{self.temp : temp})
				layer_3 = sess.run(self.update_l_s[-1], {self.temp : temp})

				# save layers 
				if liveplot:
					self.layer_save[0][step] = self.layer[0].eval()
					for layer_i in range(1,len(self.shape)-2):
						self.layer_save[layer_i][step] = layer_1[layer_i-1]
					self.layer_save[-2][step] = layer_2

				self.layer_save[-1][step] = layer_3


				if liveplot:
					# calc the energy
					self.energy_.append(sess.run(self.energy))
					# save values to array
					temp_[step] = temp
				

				# assign new temp
				temp += temp_delta 


			#### calc the probabiliy for every point in h 
			# self.p_h = sigmoid_np(np.dot(h2,self.w2_np.T)+np.dot(v_gibbs,self.w1_np), temp)
			# # calc the standart deviation for every point
			# self.std_dev_h = np.sqrt(self.p_h*(1-self.p_h))

			#### for checking of the thermal equilibrium
			# step=10
			# if i%step==0 and i>0:
			# 	self.mean_h1.append( np.mean(h1_[i-(step-1):i], axis = (0,1) ))
			# 	if len(self.mean_h1)>1:
			# 		log.out(np.mean(abs(self.std_dev_h-(abs(self.mean_h1[-2]-self.mean_h1[-1])))))
	

		if mode=="generate":
			sess.run(self.layer[-1].assign(v_input))

			for step in range(gibbs_steps):
				# update all layer except the last one 
				layer = sess.run(self.update_l_s[:-1], {self.temp : temp})

				# save layers 
				for layer_i in range(len(self.shape)-1):
					self.layer_save[layer_i][step] = layer[layer_i]
				self.layer_save[-1][step] = self.layer[-1].eval()


				if liveplot:
					# calc the energy
					self.energy_.append(sess.run(self.energy))
					# save values to array
					temp_[step] = temp
				

				# assign new temp
				temp += temp_delta 

				#### for checking of the thermal equilibrium
				# if i%100==0:
				# 	self.mean_h1.append(np.mean(h1_[i-99:i],axis=0))
				# 	if len(self.mean_h1)>1:
				# 		log.out(np.mean(abs(self.mean_h1[-2]-self.mean_h1[-1])))
	
		if liveplot and plt.fignum_exists(fig.number) and self.batchsize==1:
			data = [None]*(len(self.shape)+1)
			for layer_i in range(len(self.shape)-1):
				s = int(sqrt(self.shape[layer_i]))
				data[layer_i]  = ax[layer_i].matshow(self.layer_save[layer_i][0].reshape(s,s))
				ax[layer_i].set_yticks([])
				ax[layer_i].set_xticks([])
				ax[layer_i].grid(False)
			
			data[-2], = ax[-2].plot([],[])
			data[-1], = ax[-1].plot([],[])

			ax[0].set_title("Visible Layer")

			ax[-1].set_xlim(0,len(self.energy_))
			ax[-1].set_ylim(np.min(self.energy_),0)
			ax[-1].set_title("Energy")

			ax[-2].set_ylim(0,1)
			ax[-2].set_xlim(0,10)
			ax[-2].set_title("Classification")
			
			for step in range(1,gibbs_steps-1,2):
				if plt.fignum_exists(fig.number):
					ax[1].set_title("Temp.: %s, Steps: %s"%(str(round(temp_[step],3)),str(step)))

					for layer_i in range(len(self.shape)-1):
						s = int(sqrt(self.shape[layer_i]))
						data[layer_i].set_data(self.layer_save[layer_i][step].reshape(s,s))
					data[-2].set_data(range(10),self.layer_save[-1][step])
					data[-1].set_data(range(step),self.energy_[:step])
					
					plt.pause(1/50.)
		
			plt.close(fig)

		# return the mean of the last 20 gibbs samples for all images
		return np.mean(self.layer_save[-1][-20:,:],axis=0)

	def export(self):
		# convert weights and biases to numpy arrays
		self.w_np=[]
		for i in range(len(self.shape)-1):
			self.w_np.append(self.w[i].eval())
		self.bias_np = []
		for i in range(len(self.shape)):	
			self.bias_np.append(self.bias[i].eval())

		# convert tf.arrays to numpy arrays 
		if training:
			self.h1_activity_np = self.h1_activity_.eval()
			self.h2_activity_np = self.h2_activity_.eval()
			self.train_error_np = self.train_error_.eval()
			self.train_class_error_np = self.train_class_error_.eval()
			self.w_mean_np = []
			for i in range(len(self.shape)-1):
				self.w_mean_np.append(self.w_mean_[i].eval())
		
		self.exported = 1


	def write_to_file(self):
		if self.exported!=1:
			self.export()
		new_path = saveto_path
		os.makedirs(new_path)
		os.chdir(new_path)
		for i in range(len(self.shape)-1):
			np.savetxt("w%i.txt"%i, self.w_np[i])
		for i in range(len(self.shape)):
			np.savetxt("bias%i.txt"%i, self.bias_np[i])
		
		self.log_list.append(["train_time",self.train_time])

		with open("logfile.txt","w") as log_file:
				for i in range(len(self.log_list)):
					log_file.write(self.log_list[i][0]+","+str(self.log_list[i][1])+"\n")
		
		log.info("Saved data and log to:",new_path)

		if save_all_params:
			if training:
				np.savetxt("h1_activity.txt", self.h1_activity_np)
				np.savetxt("train_error.txt", self.train_error_np)
				np.savetxt("train_class_error.txt", self.train_class_error_np)
				np.savetxt("w1_mean.txt", self.w_mean_np[0])

			# test results
			np.savetxt("test_error_mean.txt", self.test_error[None]) 
			np.savetxt("class_error_mean.txt", self.class_error[None]) 
			np.savetxt("h1_act_test_mean.txt", self.h1_act_test[None])
			np.savetxt("h2_act_test_mean.txt", self.h2_act_test[None])
			np.savetxt("v_recon_prob_test.txt", self.probs) 
			np.savetxt("v_recon_test.txt", self.rec) 
			np.savetxt("h1_recon_test.txt", self.rec_h1) 
			np.savetxt("h1_test.txt", self.h1_test) 
			np.savetxt("h2_prob_test.txt", self.h2_test) 

			log.info("Saved Parameters to same path")


		
		os.chdir(workdir)


####################################################################################################################################
#### User Settings ###

num_batches_pretrain = 100
dbm_batches          = 500
pretrain_epochs      = [2,10,10,10,10]
dbm_epochs           = 1


rbm_learnrate     = 0.05
dbm_learnrate     = 0.005
dbm_learnrate_end = 0.005

temp = 0.05

pre_training    = 0	# if no pretrain then files are automatically loaded
training        = 1	# if trianing the whole DBM
testing         = 1	# if testing the DBM with test data
plotting        = 1

gibbs_sampling  = 0
noise_stab_test = 0

save_to_file    = 1 	# only save biases and weights for further training
save_all_params = 0	# also save all test data and reconstructed images (memory heavy)
save_pretrained = 0	


load_from_file        = 0
pathsuffix            = "Mon_Mar_26_09-24-59_2018_[784, 400, 100, 10]"#"Thu Jan 18 20-04-17 2018 80 epochen"
pathsuffix_pretrained = "Fri_Mar_23_10-22-57_2018"
####################################################################################################################################################


DBM_shape = [
			28*28,
			20*20,
			10*10,
			10
		 ]

saveto_path=data_dir+"/"+time_now+"_"+str(DBM_shape)

### modify the parameters with additional_args
if len(additional_args) > 0:
	# n_samples = int(additional_args[0])
	saveto_path    += " - "+str(additional_args[0])


######### DBM #############################################################################################
#### Pre training is ended - create the DBM with the gained weights
# if i == 0,1,2,...: (das ist das i von der echo cluster schleife) in der dbm class stehen dann die parameter fur das jeweilige i 
DBM = DBM_class(	shape = DBM_shape, liveplot = 0)

###########################################################################################################
#### Sessions ####
log.reset()
log.info(time_now)


DBM.pretrain()

for i in range(1):
	if training:
		with tf.Session() as sess:
			log.start("DBM Train Session")
			
			
			DBM.train(	train_data  = train_data,
					train_label = train_label,
					epochs      = dbm_epochs,
					num_batches = dbm_batches,
					learnrate   = dbm_learnrate,
					N           = 2, # freerunning steps
					cont        = i)

			DBM.train_time=log.end()

	# test session
	if testing:
		with tf.Session() as sess:
			log.start("Test Session")
			# wrong_classified_id = np.loadtxt("wrongs.txt").astype(np.int)

			wrong_classified_id = DBM.test(test_data, test_label,
									N = 50,  # sample ist aus random werten, also mindestens 2 sample machen 
									M = 10  # average v. 0->1 sample
								)


			# DBM.test(test_data_noise) 

			log.end()





if gibbs_sampling:
	with tf.Session() as sess:
		log.start("Gibbs Sampling Session")

		if load_from_file and not training:
			DBM.load_from_file(workdir+"/data/"+pathsuffix)

		subspace = [5,6,7,8,9]

		p = 1
		log.info("Multiplicator p = ",p)
		context_mod = np.zeros(10)
		for i in range(10):
			if i in subspace:
				context_mod[i] = 1*p

		# loop through images from all wrong classsified images and find al images that are <5 
		index_for_number_gibbs=[]
		for i in range(1000): #wrong_classified_id:			
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


		# #### generation of an image using a label
		# h2_no_context=DBM.gibbs_sampling([[1,0,0,0,0,0,0,0,0,0]], 500, 0.055 , 0.03, 
		# 					mode         = "generate",
		# 					modification = [1,1,1,1,1,1,1,1,1,1],
		# 					liveplot     = 1)

		# calculte h2 firerates over all gibbs_steps 
		log.start("Sampling data")
		h2_no_context=DBM.gibbs_sampling(test_data[index_for_number_gibbs[:]], 100, 0.05 , 0.05, 
							mode         = "context",
							modification = [1,1,1,1,1,1,1,1,1,1],
							liveplot     = 0)
			
		# # with context
		h2_context=DBM.gibbs_sampling(test_data[index_for_number_gibbs[:]], 100, 0.05 , 0.05, 
							mode         = "context",
							modification = context_mod,
							liveplot     = 0)
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

		hist_data    = np.zeros([10,1]).tolist()
		hist_data_nc = np.zeros([10,1]).tolist()

		for i,d in enumerate(index_for_number_gibbs):
			digit = np.where( test_label[d] == 1 )[0][0]
			
			hist_data[digit].append( h2_context[i].tolist() )
			hist_data_nc[digit].append( h2_no_context[i].tolist() )

			### count how many got right (with context) 
			maxi_c    = h2_context[i].max()
			max_pos_c = np.where(h2_context[i] == maxi_c)[0][0]
			if max_pos_c == digit:
				correct_maxis_c.append(maxi_c)
			else:
				if max_pos_c  not  in  subspace:
					wrongs_outside_subspace_c += 1
				incorrect_maxis_c.append(maxi_c)

			### count how many got right (no context) 
			maxi_nc    = h2_no_context[i].max()
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
		log.info("Diff:     ",abs(len(incorrect_maxis_c)-len(incorrect_maxis_nc)))
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

if noise_stab_test:
	with tf.Session() as sess:
		s=34
		my_pal=["#FF3045","#77d846","#466dd8","#ffa700","#48e8ff","#a431e5","#333333","#a5a5a5","#ecbdf9","#b1f6b6"]
		noise_h2_,v_noise=DBM.test_noise_stability(test_data[s:s+1], test_label[s:s+1])
		with seaborn.color_palette(my_pal, 10):
			for i in range(10):
				plt.plot(smooth(noise_h2_[:,i],20),label=str(i))
			plt.legend()
		plt.matshow(v_noise.reshape(28,28))

if training and save_to_file:
	DBM.write_to_file()


####################################################################################################################################
#### Plot
# Plot the Weights, Errors and other informations
h1_shape = int(sqrt(DBM.shape[1]))
if plotting:
	log.out("Plotting...")
	
	# plot w1 as image	
	map1=plt.matshow(tile(DBM.w_np[i]),cmap="gray")
	plt.colorbar(map1)
	plt.grid(False)
	plt.title("W %i"%i)

	# plot all other weights as hists
	fig,ax = plt.subplots(DBM.n_layers-1,1,figsize=(8,10))
	for i in range(DBM.n_layers-1):
		ax[i].hist((DBM.w_np[i]).flatten(),bins=60,alpha=0.5,label="Before Training")
		ax[i].hist((DBM.w_np_old[i]).flatten(),color="r",bins=60,alpha=0.5,label="After Training")
		ax[i].set_title("W %i"%i)
		ax[i].legend()
	plt.tight_layout()


	map3=plt.matshow((DBM.w_np[-1]).T)
	plt.title("W 3")
	plt.colorbar(map3)

	try:
		# plot change in w1 
		plt.matshow(tile(DBM.w_np[0]-DBM.w_np_old[0]))
		plt.colorbar()
		plt.title("Change in W1")
	except:
		pass

	# plot the layer_act for 100 pictures
	plt.figure("Layer_activiations_test_run")
	for i in range(self.n_layers):
		plt.plot(self.layer_act[:,i],label="Layer %i"%i)
	plt.legend()


	# timeline
	fig,ax=plt.subplots(2,len(DBM.image_timeline),figsize=(17,6))
	plt.tight_layout()
	for i in range(len(DBM.image_timeline)):
		ax[0][i].matshow((DBM.image_timeline[i]).reshape(28,28))
		ax[1][i].matshow((DBM.save_h1[i*2]).reshape(int(sqrt(DBM.shape[1])),int(sqrt(DBM.shape[1]))))
		ax[0][i].set_title(str(i))
		ax[0][i].set_xticks([])
		ax[0][i].set_yticks([])
		ax[1][i].set_xticks([])
		ax[1][i].set_yticks([])
		ax[0][i].grid(False)
	


	if training:
		x=np.linspace(0,dbm_epochs,len(DBM.w_mean_np[0]))

		fig_fr=plt.figure(figsize=(7,9))
		
		ax_fr1=fig_fr.add_subplot(311)
		ax_fr1.plot(x,DBM.h1_activity_np)
		
		ax_fr2=fig_fr.add_subplot(312)
		# ax_fr2.plot(DBM.CD1_mean_np,label="CD1")
		# ax_fr2.plot(DBM.CD2_mean_np,label="CD2")
		for i in range(len(DBM.shape)-1):
			ax_fr2.plot(x,DBM.w_mean_np[i],label="Weights %i"%i)
		ax_fr1.set_title("Firerate h1 layer")
		ax_fr2.set_title("Weights mean")
		ax_fr2.legend(loc="best")
		# ax_fr2.set_ylim([0,np.max(DBM.w_mean_np[0])*1.1])
		ax_fr3=fig_fr.add_subplot(313)
		ax_fr3.plot(x,DBM.train_error_np,"k",label="Reconstruction")
		ax_fr3.plot(x,DBM.train_class_error_np,"r",label="Classification")
		plt.legend(loc="best")
		ax_fr3.set_title("Train Error")
		
	plt.tight_layout()


	#plot some samples from the testdata 
	fig3,ax3 = plt.subplots(len(DBM.shape)+1,13,figsize=(16,6),sharey="row")
	plt.tight_layout(pad=0.0)
	for i in range(13):
		# plot the input
		ax3[0][i].matshow(test_data[i:i+1].reshape(int(sqrt(DBM.shape[0])),int(sqrt(DBM.shape[0]))))
		ax3[0][i].set_yticks([])
		ax3[0][i].set_xticks([])
		# plot the reconstructed image		
		ax3[1][i].matshow(DBM.probs[i:i+1].reshape(int(sqrt(DBM.shape[0])),int(sqrt(DBM.shape[0]))))
		ax3[1][i].set_yticks([])
		ax3[1][i].set_xticks([])
		
		#plot all layers that can get imaged
		for layer in range(len(DBM.shape)-2):
			ax3[layer+2][i].matshow(DBM.hidden_save[layer][i:i+1].reshape(int(sqrt(DBM.shape[layer+1])),int(sqrt(DBM.shape[layer+1]))))
			ax3[layer+2][i].set_yticks([])
			ax3[layer+2][i].set_xticks([])

		# plot the last layer 		
		ax3[-1][i].bar(range(10),DBM.label_test[i])
		ax3[-1][i].set_xticks(range(10))
		ax3[-1][i].set_ylim(0,1)

		#plot the reconstructed layer h1
		# ax3[5][i].matshow(DBM.rec_h1[i:i+1].reshape(int(sqrt(DBM.shape[1])),int(sqrt(DBM.shape[1]))))
		# plt.matshow(random_recon.reshape(28,28))
	


	#plot only one digit
	fig3,ax3 = plt.subplots(len(DBM.shape)+1,10,figsize=(16,6),sharey="row")
	m=0
	for i in index_for_number_test[0:10]:
		# plot the input
		ax3[0][m].matshow(test_data[i:i+1].reshape(int(sqrt(DBM.shape[0])),int(sqrt(DBM.shape[0]))))
		ax3[0][m].set_yticks([])
		ax3[0][m].set_xticks([])
		# plot the reconstructed image		
		ax3[1][m].matshow(DBM.probs[i:i+1].reshape(int(sqrt(DBM.shape[0])),int(sqrt(DBM.shape[0]))))
		ax3[1][m].set_yticks([])
		ax3[1][m].set_xticks([])
		
		#plot all layers that can get imaged
		for layer in range(len(DBM.shape)-2):
			ax3[layer+2][m].matshow(DBM.hidden_save[layer][i:i+1].reshape(int(sqrt(DBM.shape[layer+1])),int(sqrt(DBM.shape[layer+1]))))
			ax3[layer+2][m].set_yticks([])
			ax3[layer+2][m].set_xticks([])

		# plot the last layer 		
		ax3[-1][m].bar(range(10),DBM.label_test[i])
		ax3[-1][m].set_xticks(range(10))
		ax3[-1][m].set_ylim(0,1)
		#plot the reconstructed layer h1
		# ax4[5][m].matshow(DBM.rec_h1[i:i+1].reshape(int(sqrt(DBM.shape[1])),int(sqrt(DBM.shape[1]))))
		# plt.matshow(random_recon.reshape(28,28))
		m+=1
	plt.tight_layout(pad=0.0)




plt.show()
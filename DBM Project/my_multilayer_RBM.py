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

		self.w       = tf.Variable(tf.random_uniform([self.visible_units,self.hidden_units],minval=-1e-3,maxval=1e-3),name="Weights")# init with small random values to break symmetriy
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

		self.RBMs    = [0]*(len(self.shape)-1)
		self.RBMs[0] = RBM(self.shape[0],self.shape[1], forw_mult= 1, back_mult = 1, learnrate = rbm_learnrate, liveplot=0)
		self.RBMs[1] = RBM(self.shape[1],self.shape[2], forw_mult= 1, back_mult = 1, learnrate = rbm_learnrate, liveplot=0)
		self.RBMs[2] = RBM(self.shape[2],self.shape[3], forw_mult= 1, back_mult = 1, learnrate = rbm_learnrate, liveplot=0)

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
						np.savetxt("Pretrained-"+" %i "%i+str(time_now)+".txt", self.weights[i])
					log.out("Saved Pretrained under "+str(time_now))
			else:
				self.weights=[]
				log.out("Loading Pretrained from file")
				for i in range(len(self.shape)-1):
					self.weights.append(np.loadtxt("Pretrained-"+" %i "%i+pathsuffix_pretrained+".txt").astype(np.float32))

			log.end()


	def load_from_file(self,path):
		os.chdir(path)
		log.out("Loading data from:","...",path[-20:])
		self.w1_np     = np.loadtxt("w1.txt")
		self.w1_np_old = self.w1_np #save weights for later comparison
		self.w2_np     = np.loadtxt("w2.txt")
		self.w3_np     = np.loadtxt("w3.txt")
		self.bias1_np  = np.loadtxt("bias1.txt")
		self.bias2_np  = np.loadtxt("bias2.txt")
		self.bias3_np  = np.loadtxt("bias3.txt")
		self.bias_label_np  = np.loadtxt("bias_label.txt")
		os.chdir(workdir)

	
	def import_(self):
		""" setting up the graph and setting the weights and biases tf variables to the 
		saved numpy arrays """
		log.out("loading numpy vars into graph")
		sess.run(self.w1.assign(self.w1_np))
		sess.run(self.w2.assign(self.w2_np))
		sess.run(self.w3.assign(self.w3_np))
		sess.run(self.bias_v.assign(self.bias1_np))
		sess.run(self.bias_h1.assign(self.bias2_np))
		sess.run(self.bias_h2.assign(self.bias3_np))
		sess.run(self.bias_label.assign(self.bias_label_np))

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
		

		self.v  = tf.placeholder(tf.float32,[self.batchsize,self.shape[0]],name="Visible-Layer") # has self.shape [number of images per batch,number of visible units]

		self.batch_ph       = tf.placeholder(tf.float32,[self.batchsize,self.shape[0]],name="Batch_placeholder")
		self.batch_label_ph = tf.placeholder(tf.float32,[self.batchsize,self.shape[-1]],name="Batchlabel_placeholder")
		self.h1_ph          = tf.placeholder(tf.float32,[self.batchsize,self.shape[1]],name="h1_placeholder")
		self.h2_ph          = tf.placeholder(tf.float32,[self.batchsize,self.shape[2]],name="h2_placeholder")

		if graph_mode=="training":
			# h2 and other stuff is plaveholder if training
			self.m_tf      = tf.placeholder(tf.int32,[],name="running_array_index")
			self.h2        = tf.placeholder(tf.float32,[self.batchsize,self.shape[2]],name="placeholder_h2")
			self.learnrate = tf.placeholder(tf.float32,[],name="Learnrate")

			# arrays for saving progress
			self.h1_activity_ = tf.Variable(tf.zeros([self.num_of_updates/self.num_of_skipped]))
			self.h2_activity_ = tf.Variable(tf.zeros([self.num_of_updates/self.num_of_skipped]))
			self.train_error_ = tf.Variable(tf.zeros([self.num_of_updates/self.num_of_skipped]))
			self.train_class_error_ = tf.Variable(tf.zeros([self.num_of_updates/self.num_of_skipped]))
			self.w1_mean_     = tf.Variable(tf.zeros([self.num_of_updates/self.num_of_skipped]))

			# gradients for update of Weights
			self.pos_grad1 = tf.Variable(tf.zeros([self.shape[0],self.shape[1]]))
			self.neg_grad1 = tf.Variable(tf.zeros([self.shape[0],self.shape[1]]))

			self.pos_grad2 = tf.Variable(tf.zeros([self.shape[1],self.shape[2]]))
			self.neg_grad2 = tf.Variable(tf.zeros([self.shape[1],self.shape[2]]))

			self.pos_grad3 = tf.Variable(tf.zeros([self.shape[2],self.shape[3]]))
			self.neg_grad3 = tf.Variable(tf.zeros([self.shape[2],self.shape[3]]))

		#### create vars for each layer that get assigned for sampling
		self.v_var   = tf.Variable(tf.random_uniform([self.batchsize,self.shape[0]],minval=-1e-3,maxval=1e-3),name="v_var")

		self.h1_var     = tf.Variable(tf.random_uniform([self.batchsize,self.shape[1]],minval=-1e-3,maxval=1e-3),name="h1_var")
		self.h1_var_old = tf.Variable(tf.random_uniform([self.batchsize,self.shape[1]],minval=-1e-3,maxval=1e-3),name="h1_var_old")
		self.h2_var     = tf.Variable(tf.random_uniform([self.batchsize,self.shape[2]],minval=-1e-3,maxval=1e-3),name="h2_var")
		self.h2_var_old = tf.Variable(tf.random_uniform([self.batchsize,self.shape[2]],minval=-1e-3,maxval=1e-3),name="h2_var_old")
		
		self.label_var  = tf.Variable(tf.random_uniform([self.batchsize,self.shape[-1]],minval=-1e-3,maxval=1e-3),name="label_var")
		

		# modification array size 10 that gehts multiplied to the label vector for context
		self.modification_tf = tf.Variable(tf.ones([self.batchsize,self.shape[-1]]),name="Modification")

		# init vars with batches
		self.assign_v  = self.v_var.assign(self.sample(self.batch_ph))
		
		self.assign_h1 = self.h1_var.assign(self.h1_ph)
		self.assign_h2 = self.h2_var.assign(self.h2_ph)

		self.assign_label = self.label_var.assign(self.batch_label_ph)

		
		#### temperature
		if graph_mode=="gibbs" or graph_mode=="testing":
			self.temp = tf.placeholder(tf.float32,[],name="Temperature")
		else:
			self.temp = temp


		### Parameters 
		self.w1 = tf.Variable(self.weights[0],name="Weights1")
		self.w2 = tf.Variable(self.weights[1],name="Weights2")
		self.w3 = tf.Variable(self.weights[2],name="Weights3")

		self.bias_v     = tf.Variable(tf.zeros([self.shape[0]]),name="Visible-Bias")
		self.bias_h1    = tf.Variable(tf.zeros([self.shape[1]]), name="Hidden-Bias")
		self.bias_h2    = tf.Variable(tf.zeros([self.shape[2]]), name="Hidden2-Bias")
		self.bias_label = tf.Variable(tf.zeros([self.shape[-1]]), name="label-Bias")


		### Error and stuff
		self.error       = tf.reduce_mean(tf.square(self.batch_ph-self.v_var))
		self.class_error = tf.reduce_mean(tf.square(self.batch_label_ph-self.label_var))

		self.h1_sum = tf.reduce_sum(self.h1_var)
		self.h2_sum = tf.reduce_sum(self.h2_var)
		self.label_sum = tf.reduce_sum(self.label_var)

		self.free_energy = -tf.reduce_sum(tf.log(1+tf.exp(tf.matmul(self.v_var,self.w1)+self.bias_h1)))

		## calculations for layers
		self.v_prob = sigmoid(tf.matmul(self.h1_var, self.w1,transpose_b=True)+self.bias_v,self.temp)
		self.v = self.sample(self.v_prob)

		self.h1_prob = sigmoid(tf.matmul(self.v_var,self.w1)  + tf.matmul(self.h2_var,self.w2,transpose_b=True) + self.bias_h1, self.temp)
		self.h1 = self.sample(self.h1_prob)
		
		self.h2_prob = sigmoid(tf.matmul(self.h1_var,self.w2) + tf.matmul(self.label_var,self.w3,transpose_b=True) + self.bias_h2, self.temp)
		self.h2 = self.sample(self.h2_prob)

		self.label_prob = sigmoid(tf.matmul(self.h2_var, self.w3) + self.bias_label, self.temp)
		self.label = self.sample(self.label_prob)

		#### updates for each layer 
		self.update_h2_with_context = self.h2_var.assign(self.sample(sigmoid(tf.matmul(self.h1_var,self.w2)  
											+ tf.matmul(tf.multiply(self.label_var,self.modification_tf),self.w3,transpose_b=True)
											+ self.bias_h2,self.temp)))
		self.update_h1_probs = self.h1_var.assign(self.h1_prob)			
		self.update_h2_probs = self.h2_var.assign(self.h2_prob)			
		
		self.update_v_probs = self.v_var.assign(self.v_prob)			

		self.update_v  = self.v_var.assign(self.v)
		self.update_h1 = self.h1_var.assign(self.h1)		
		self.update_h2 = self.h2_var.assign(self.h2)		
		self.update_label = self.label_var.assign(self.label_prob)
		
		self.update_all_layer = [
						self.update_v,
						self.update_h1,
						self.update_h2,
						self.update_label
						]


		self.numpoints       = tf.cast(tf.shape(self.v_var)[0],tf.float32) #number of train inputs per batch (for averaging the CD matrix -> see practical paper by hinton)

		### Training with contrastive Divergence
		if graph_mode=="training":

			### def weigth update (w,v,h1,v',h1'): tf.assign(w, <v*h1> - <v'*h1'>)

			#first weight matrix
			self.update_pos_grad1 = self.pos_grad1.assign(tf.matmul(self.v_var, self.h1_var,transpose_a=True))
			self.update_neg_grad1 = self.neg_grad1.assign(tf.matmul(self.v_var, self.h1_var,transpose_a=True))
			self.CD1              = ((self.pos_grad1 - self.neg_grad1))/self.numpoints
			self.update_w1        = self.w1.assign_add(tf.multiply(self.learnrate,self.CD1))
			self.mean_w1          = tf.reduce_mean(tf.square(self.w1))
			
			# second weight matrix
			self.update_pos_grad2 = self.pos_grad2.assign(tf.matmul(self.h1_var, self.h2,transpose_a=True))
			self.update_neg_grad2 = self.neg_grad2.assign(tf.matmul(self.h1_var, self.h2,transpose_a=True))
			self.CD2              = ((self.pos_grad2 - self.neg_grad2))/self.numpoints
			self.update_w2        = self.w2.assign_add(tf.multiply(self.learnrate,self.CD2))
			
			#third weight matrix
			self.update_pos_grad3 = self.pos_grad3.assign(tf.matmul(self.h2, self.label_var,transpose_a=True))
			self.update_neg_grad3 = self.neg_grad3.assign(tf.matmul(self.h2, self.label_var,transpose_a=True))
			self.CD3              = ((self.pos_grad3 - self.neg_grad3))/self.numpoints
			self.update_w3        = self.w3.assign_add(tf.multiply(self.learnrate,self.CD3))

			# bias updates
			self.update_h1_old  = self.h1_var_old.assign(self.h1_var)
			self.update_h2_old  = self.h2_var_old.assign(self.h2_var)

			self.update_all_bias = [self.bias_h1.assign_add(tf.multiply(self.learnrate,tf.reduce_mean(tf.subtract(self.h1_var_old,self.h1_var),0))),
							self.bias_h2.assign_add(tf.multiply(self.learnrate,tf.reduce_mean(tf.subtract(self.h2_var_old,self.h2_var),0))),
							self.bias_v.assign_add(tf.multiply(self.learnrate,tf.reduce_mean(tf.subtract(self.batch_ph,self.v_var),0))),
							self.bias_label.assign_add(tf.multiply(self.learnrate,tf.reduce_mean(tf.subtract(self.batch_label_ph,self.label_var),0)))
							]
		
			self.assign_arrays =	[tf.scatter_update(self.train_error_,self.m_tf,self.error),
							 tf.scatter_update(self.train_class_error_,self.m_tf,self.class_error),
							 tf.scatter_update(self.w1_mean_,self.m_tf,self.mean_w1),
							 tf.scatter_update(self.h1_activity_,self.m_tf,self.h1_sum),
							]
		

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

	# def freerun(self,v_input,N):
	# 	self.batchsize = len(v_input)
	# 	v = np.zeros([N,self.batchsize,self.shape[0]])
	# 	h1 = np.zeros([N,self.batchsize,self.shape[1]])
	# 	h2 = np.zeros([N,self.batchsize,self.shape[2]])

	# 	self.v_var.assign(v_input.reshape(1,28))

	# 	for n in range(N):
	# 		v[n], h1[n], h2[n] = sess.run([self.update_v,self.update_h1,self.update_h2])

	# 	fig,ax = plt.subplots(1,10)
	# 	for i in range(10):
	# 		ax[i].matshow(v[i].reshape(28,28))



	def train(self,train_data,train_label,epochs,num_batches,learnrate,N,cont):
		""" training the DBM with given h2 as labels and v as input images
		train_data :: images
		train_label :: corresponding label
		epochs :: how many epochs to train
		num_batches :: how many batches
		learnrate :: learnrate
		N :: Number of gibbs steps
		M :: Number of particles (how many times to gibbs sample)
		"""
		######## init all vars for training
		self.batchsize      = int(55000/num_batches)
		num_of_updates      = epochs*num_batches
		
		log.info("Batchsize:",self.batchsize,"NBatches",num_of_updates)

		self.num_of_updates = num_of_updates
		d_learnrate         = float(dbm_learnrate_end-learnrate)/num_of_updates
		self.m              = 0

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
				sess.run([self.assign_v,self.assign_label],{ self.batch_ph : batch, 
										 	self.batch_label_ph : batch_label})

				# calc hidden layer probabilities
				for hidden in range(10):
					sess.run([self.update_h1_probs,self.update_h2_probs])
					
				# save this for bias update
				sess.run([self.update_h1_old, self.update_h2_old])

				# update the positive gradients
				sess.run([self.update_pos_grad1,self.update_pos_grad2,self.update_pos_grad3])


				# update all layers N times (free running, gibbs sampling) 
				for n in range(N):
					sess.run(self.update_all_layer)
				# las step to reduce some sampling noise update h1 but only calc probs
				sess.run([self.update_h1_probs,self.update_h2_probs])
						
				


				# calc he negatie gradients
				sess.run([self.update_neg_grad1,self.update_neg_grad2,self.update_neg_grad3])


				# run all parameter updates 
				sess.run([	self.update_w1,
						self.update_w2,
						self.update_w3, #weight update als function schreiben?
						self.update_all_bias
						],
						feed_dict={	self.batch_ph : batch,
								self.batch_label_ph : batch_label,
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
								feed_dict={	self.batch_ph : batch,
										self.batch_label_ph : batch_label,
										self.m_tf:self.m / self.num_of_skipped}
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
		self.h1_activity_*=1./(n_second_layer*self.batchsize)

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

		h1    = np.zeros([N,self.batchsize,self.shape[1]])
		h2    = np.zeros([N,self.batchsize,self.shape[2]])
		label = np.zeros([N,self.batchsize,self.shape[-1]])

		self.label_diff = np.zeros([N,self.batchsize,self.shape[-1]])


		### init the graph 
		if load_from_file and not training:
			self.load_from_file(workdir+"/data/"+pathsuffix)
		self.graph_init("testing") # "testing" because this graph creates the testing variables where only v is given, not h2
		self.import_()


		#### start test run
		log.start("Testing DBM with %i images"%self.batchsize)
		
		#### give input to v layer
		sess.run(self.assign_v, {self.batch_ph : test_data, self.temp : temp})

		#### update hidden and label N times
		log.out("Sampling h1 and h2 %i times"%N)
		for n in range(N):
			h1[n], h2[n], label[n]  = sess.run([self.update_h1, self.update_h2, self.update_label], {self.temp : temp})
			### calculate diffs vs the N steps 
			if n>0:
				self.label_diff[n-1] = np.abs(label[n]-label[n-1])


		# plot the diffst for 100 pictures
		if plotting:
			diffs_label_plt=[]
			save = np.zeros(N)
			for pic in range(100):
				for i in range(N):
					save[i]=np.mean(DBM.label_diff[i,pic,:])
				diffs_label_plt.append(smooth(save,10))
				plt.plot(diffs_label_plt[pic])
			plt.xlabel("N")
			plt.title("differenzen der label layer fur 100 bilder")

		
		self.h1_test = np.mean(h1[-20:],axis=0)
		self.label_test = np.mean(label[-20:],axis=0)

		#### update v M times
		self.probs = self.v_var.eval()
		for i in range(M):
			self.probs += sess.run(self.update_v,{self.temp : temp})
		self.probs *= 1./(M+1)



		#### calculate errors and activations
		self.test_error  = self.error.eval({self.batch_ph : test_data})
		self.test_error_.append(self.test_error) #append to errors if called multiple times
		# error of classifivation labels
		self.class_error=np.mean(np.abs(self.label_test-test_label))		
		#activations of hidden layers
		self.h1_act_test = self.h1_sum.eval()
		self.label_act_test = self.label_sum.eval()
		# norm the sum of the activities
		self.h1_act_test*=1./(n_second_layer*len(test_data))
		self.label_act_test*=1./(n_third_layer*len(test_data))

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

						# label_copy[i]=self.label_test[i]==self.label_test[i].max()
						# if this sum == 1 its a wrong classification
						# sum_=np.sum(label_copy[i]!=test_label[i])/2
						# n_wrongs+=sum_
						## search which numbers got not classified correctly
						# if sum_==1:
						# 	wrong_classified_ind.append(i)


		log.end()
		log.reset()
		log.info("Reconstr. error: ",np.round(DBM.test_error,5), "learnrate: ",np.round(dbm_learnrate,5))
		log.info("Class error: ",np.round(self.class_error,5))
		log.info("Wrong Digits: ",n_wrongs," with average: ",round(np.mean(wrong_maxis),3))
		log.info("Activations of Neurons: ", np.round(self.h1_act_test,4) , np.round(self.label_act_test,4))
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

		# self.v_rev = tf.Variable(tf.random_uniform([1,self.shape[0]],minval=-1e-3,maxval=1e-3),name="v_rev_init")
		# self.h2    = tf.Variable(tf.random_uniform([1,self.shape[2]],minval=-1e-3,maxval=1e-3),name="h2")
		
		# tf.variables_initializer([self.h2,self.v_rev], name='init_train')

		h2_            = np.zeros([gibbs_steps,self.batchsize,self.shape[2]])
		h1_            = np.zeros([gibbs_steps,self.batchsize,self.shape[1]])
		v_g_           = np.zeros([gibbs_steps,self.batchsize,self.shape[0]])
		temp_          = np.zeros([gibbs_steps])
		self.energy_   = []
		self.mean_h1   = []
		temp           = temp_start
		temp_delta     = (temp_end-temp_start)/gibbs_steps

		self.num_of_updates=1000 #just needs to be defined because it will make a train graph with tf.arrays where this number is needed


		if liveplot:
			log.info("Liveplotting gibbs sampling")
			fig,ax=plt.subplots(1,4,figsize=(15,5))



		if mode=="context":
			# init the v layer and h1 layer
			sess.run(self.assign_v, {self.batch_ph : v_input})
			sess.run(self.assign_h1, {self.h1_ph : rnd.random([self.batchsize,self.shape[1]])})
			sess.run(self.assign_h2, {self.batch_label_ph : rnd.random([self.batchsize,self.shape[2]])})
			# sess.run(self.label_var.assign(np.reshape(modification,[1,10])))
			
			h2 = self.label_var.eval()
			h1 = self.h1_var.eval()
			v_gibbs = self.v_var.eval()

			modification = np.concatenate((modification,)*self.batchsize).reshape(self.batchsize,10)
			sess.run(self.modification_tf.assign(modification))
			
			for i in range(gibbs_steps):
				# calculate the backward and forward pass 
				h1, h2 = sess.run([self.update_h1_with_context, self.update_h2], {self.temp: temp})
				# v_gibbs = sess.run(self.update_v, {self.temp : temp})
				h2_[i] = h2


				if liveplot and self.batchsize==1:
					# calc the energy
					energy1=np.dot(v_gibbs, np.dot(self.w1_np,h1.T))[0]
					energy2=np.dot(h1, np.dot(self.w2_np,h2.T))[0]
					self.energy_.append(-(energy1+energy2))
					# save values to array
					
					h1_[i]   = h1
					v_g_[i]  = v_gibbs
					temp_[i] = temp
				

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
			sess.run(self.label_var.assign(v_input))
			sess.run(self.assign_v,{self.batch_ph : rnd.random([1,DBM.shape[0]])*0.1})
			sess.run(self.update_h1, {self.temp : temp})
			h2 = self.label_var.eval()
			h1 = self.h1_var.eval()
			v_gibbs = self.v_var.eval()

			# sess.run(self.modification_tf.assign(self.batchsize*[modification]))
			
			# log.info("Generating with image as starting value for v")
			# input_image = test_data[3:4]
			## make noisy ?

			for i in range(gibbs_steps):
				
				# calculate the backward and forward pass 
				v_gibbs, h1 = sess.run([self.update_v, self.update_h1], {self.temp : temp})

				h2_[i] = h2 # sess.run(self.update_h2, {self.temp : temp})


				if liveplot:
					# calc the energy
					energy1=np.dot(v_gibbs, np.dot(self.w1_np,h1.T))[0]
					energy2=np.dot(h1, np.dot(self.w2_np,h2.T))[0]
					self.energy_.append(-(energy1+energy2))
					# save values to array
					
					h1_[i]   = h1
					v_g_[i]  = v_gibbs
					temp_[i] = temp
				

				# assign new temp
				temp += temp_delta 

				#### for checking of the thermal equilibrium
				# if i%100==0:
				# 	self.mean_h1.append(np.mean(h1_[i-99:i],axis=0))
				# 	if len(self.mean_h1)>1:
				# 		log.out(np.mean(abs(self.mean_h1[-2]-self.mean_h1[-1])))
	

		if liveplot and plt.fignum_exists(fig.number) and self.batchsize==1:
			a0 = ax[0].matshow(v_g_[0].reshape(28,28))

			a2=ax[2].matshow(h1_[0].reshape(int(sqrt(self.shape[1])),int(sqrt(self.shape[1]))))
			a1, = ax[1].plot(range(10),h2_[i][0],"-o")
			a3, = ax[3].plot([],[])
			ax[0].set_title("Visible Layer")
			ax[2].set_title("Hidden Layer")
			ax[2].set_yticks([])
			ax[2].set_xticks([])
			ax[0].set_yticks([])
			ax[0].set_xticks([])

			ax[3].set_xlim(0,len(self.energy_))
			ax[3].set_ylim(np.min(self.energy_),0)
			ax[3].set_title("Energy")

			ax[1].set_ylim(0,1)
			ax[1].set_title("Classification")
			ax[1].grid()
			ax[1].set_title("Temp.: %s, Steps: %s"%(str(round(temp_[0],3)),str(i)))
			
			for i in range(1,len(h2_),2):
				if plt.fignum_exists(fig.number):
					ax[1].set_title("Temp.: %s, Steps: %s"%(str(round(temp_[i],3)),str(i)))
					
					a0.set_data(v_g_[i].reshape(28,28))
					a1.set_data(range(10),h2_[i][0])
					

					a2.set_data(h1_[i].reshape(int(sqrt(self.shape[1])),int(sqrt(self.shape[1]))))
					
					a3.set_data(range(i),self.energy_[:i])
					plt.pause(1/50.)
		
			plt.close(fig)

		# return the mean of the last 20 gibbs samples for all images
		return np.mean(h2_[-20:,:,:],axis=0)

	def export(self):
		# convert weights and biases to numpy arrays
		self.w1_np    = self.w1.eval() 
		self.w2_np    = self.w2.eval()
		self.w3_np    = self.w3.eval()
		self.bias1_np = self.bias_v.eval()
		self.bias2_np = self.bias_h1.eval()
		self.bias3_np = self.bias_h2.eval()
		self.bias_label_np = self.bias_label.eval()

		# convert tf.arrays to numpy arrays 
		if training:
			self.h1_activity_np = self.h1_activity_.eval()
			self.h2_activity_np = self.h2_activity_.eval()
			self.train_error_np = self.train_error_.eval()
			self.train_class_error_np = self.train_class_error_.eval()
			self.w1_mean_np     = self.w1_mean_.eval()
			# self.CD1_mean_np    = self.CD1_mean_.eval()
			# self.CD2_mean_np    = self.CD2_mean_.eval()
		
		self.exported = 1


	def write_to_file(self):
		if self.exported!=1:
			self.export()
		new_path = saveto_path
		os.makedirs(new_path)
		os.chdir(new_path)
		np.savetxt("w1.txt", self.w1_np)
		np.savetxt("w2.txt", self.w2_np)
		np.savetxt("w3.txt", self.w3_np)
		np.savetxt("bias1.txt", self.bias1_np)
		np.savetxt("bias2.txt", self.bias2_np)
		np.savetxt("bias3.txt", self.bias3_np)
		np.savetxt("bias_label.txt", self.bias_label_np)
		
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
				np.savetxt("w1_mean.txt", self.w1_mean_np)

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
dbm_batches          = 1000 
pretrain_epochs      = [3,5,5]
dbm_epochs           = 5


rbm_learnrate     = 0.05
dbm_learnrate     = 0.05
dbm_learnrate_end = 0.05


temp = 0.05


pre_training    = 0 	#if no pretrain then files are automatically loaded


training        = 1

testing         = 1
plotting        = 1

gibbs_sampling  = 0
noise_stab_test = 0


save_to_file          = 0 	# only save biases and weights for further training
save_all_params       = 0	# also save all test data and reconstructed images (memory heavy)
save_pretrained       = 0


load_from_file        = 1
pathsuffix            = r"Sun_Mar_18_15-44-33_2018"#"Sun Feb 11 20-20-39 2018"#"Thu Jan 18 20-04-17 2018 80 epochen"
pathsuffix_pretrained = "Sun_Mar_18_15-44-33_2018"



n_first_layer    = 784
n_second_layer   = 14*14
n_third_layer    = 5*5
n_fourth_layer   = 10

saveto_path=data_dir+"/"+time_now

### modify the parameters with additional_args
if len(additional_args)>0:
	# n_samples = int(additional_args[0])
	saveto_path    += " - "+str(additional_args[0])


######### DBM ##########################################################################
#### Pre training is ended - create the DBM with the gained weights
# if i == 0,1,2,...: (das ist das i von der echo cluster schleife) in der dbm class stehen dann die parameter fur das jeweilige i 
DBM = DBM_class(	shape    = [n_first_layer,n_second_layer,n_third_layer,n_fourth_layer],
			liveplot = 0
			)

####################################################################################################################################
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
					N           = 10, # freerunning steps
					cont        = i)

			DBM.train_time=log.end()

	# test session
	if testing:
		with tf.Session() as sess:
			log.start("Test Session")
			# wrong_classified_id = np.loadtxt("wrongs.txt").astype(np.int)

			wrong_classified_id = DBM.test(test_data, test_label,
									N = 50,  # sample h1 and h2. 1->1 sample, aber achtung_> 1. sample ist aus random werten, also mindestens 2 sample machen 
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
		for i in range(10000): #wrong_classified_id:			
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
		# h2_no_context=DBM.gibbs_sampling([[0,0,1,0,0,0,0,0,0,0]], 500, 0.05 , 0.05, 
		# 					mode         = "generate",
		# 					modification = [1,1,1,1,1,1,1,1,1,1],
		# 					liveplot     = 1)

		# calculte h2 firerates over all gibbs_steps 
		log.start("Sampling data")
		h2_no_context=DBM.gibbs_sampling(test_data[index_for_number_gibbs[:]], 100, 0.05 , 0.05, 
							mode         = "context",
							modification = [1,1,1,1,1,1,1,1,1,1],
							liveplot     = 0)
			
		# with context
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
h1_shape = int(sqrt(n_second_layer))
if plotting:
	log.out("Plotting...")

		
	map1=plt.matshow(tile(DBM.w1_np),cmap="gray")
	plt.colorbar(map1)
	plt.grid(False)
	plt.title("W 1")

	# plt.matshow(tile(DBM.CD1_np))
	map2=plt.matshow(tile(DBM.w2_np))
	plt.title("W 2")
	plt.colorbar(map2)	

	try:
		# plot change in w1 
		plt.matshow(tile(DBM.w1_np-DBM.w1_np_old))
		plt.colorbar()
		plt.title("Change in W1")
	except:
		pass

	if training:
		x=np.linspace(0,dbm_epochs,len(DBM.w1_mean_np))

		fig_fr=plt.figure(figsize=(7,9))
		
		ax_fr1=fig_fr.add_subplot(311)
		ax_fr1.plot(x,DBM.h1_activity_np)
		
		ax_fr2=fig_fr.add_subplot(312)
		# ax_fr2.plot(DBM.CD1_mean_np,label="CD1")
		# ax_fr2.plot(DBM.CD2_mean_np,label="CD2")
		ax_fr2.plot(x,DBM.w1_mean_np,label="Weights")
		ax_fr1.set_title("Firerate h1 layer")
		ax_fr2.set_title("Weights mean")
		ax_fr2.legend(loc="best")
		ax_fr2.set_ylim([0,np.max(DBM.w1_mean_np)*1.1])
		ax_fr3=fig_fr.add_subplot(313)
		ax_fr3.plot(x,DBM.train_error_np,"k",label="Reconstruction")
		ax_fr3.plot(x,DBM.train_class_error_np,"r",label="Classification")
		plt.legend(loc="best")
		ax_fr3.set_title("Train Error")
		
		plt.tight_layout()


	#plot some samples from the testdata 
	fig3,ax3 = plt.subplots(4,13,figsize=(16,4),sharey="row")
	for i in range(13):
		# plot the input
		ax3[0][i].matshow(test_data[i:i+1].reshape(28,28))
		ax3[0][i].set_yticks([])
		ax3[0][i].set_xticks([])
		# plot the probs of visible layer
		ax3[1][i].matshow(DBM.probs[i:i+1].reshape(28,28))
		ax3[1][i].set_yticks([])
		ax3[1][i].set_xticks([])
		
		ax3[2][i].matshow(DBM.h1_test[i:i+1].reshape(int(sqrt(DBM.shape[1])),int(sqrt(DBM.shape[1]))))
		ax3[2][i].set_yticks([])
		ax3[2][i].set_xticks([])
		
		ax3[3][i].bar(range(10),DBM.label_test[i])
		ax3[3][i].set_xticks(range(10))

		#plot the reconstructed layer h1
		# ax3[5][i].matshow(DBM.rec_h1[i:i+1].reshape(int(sqrt(DBM.shape[1])),int(sqrt(DBM.shape[1]))))
		# plt.matshow(random_recon.reshape(28,28))
	plt.tight_layout(pad=0.0)


	#plot only one digit
	fig4,ax4 = plt.subplots(4,10,figsize=(16,4),sharey="row")
	m=0
	for i in index_for_number_test[0:10]:
		# plot the input
		ax4[0][m].matshow(test_data[i:i+1].reshape(28,28))
		ax4[0][m].set_yticks([])
		ax4[0][m].set_xticks([])
		# plot the probs of visible layer
		ax4[1][m].matshow(DBM.probs[i:i+1].reshape(28,28))
		ax4[1][m].set_yticks([])
		ax4[1][m].set_xticks([])
		# plot the hidden layer h2 and h1
		ax4[2][m].matshow(DBM.h1_test[i:i+1].reshape(int(sqrt(DBM.shape[1])),int(sqrt(DBM.shape[1]))))
		ax4[1][m].set_yticks([])
		ax4[1][m].set_xticks([])

		ax4[3][m].bar(range(10),DBM.label_test[i])
		ax4[3][m].set_xticks(range(10))
		#plot the reconstructed layer h1
		# ax4[5][m].matshow(DBM.rec_h1[i:i+1].reshape(int(sqrt(DBM.shape[1])),int(sqrt(DBM.shape[1]))))
		# plt.matshow(random_recon.reshape(28,28))
		m+=1
	plt.tight_layout(pad=0.0)


	################################################################
	# testing gibbs sampling over many modifications of the h2 layer
	# p=[0.1,0.5,1,1.5,2,3,4,5,6,8,10,12.5,15,20] #these numbers got multiplied with the modification array, modification was [1,1,1,1,1,0,0,0,0,0]
	# # probabilities with context
	# desired_digits_over_p = [0.8064, 0.8216, 0.8388, 0.8558, 0.8699, 0.9060, 0.9421, 0.9562, 0.9699, 0.9819, 0.9937, 0.9976, 0.9982, 0.998]
	# wrong_digits_over_p   = [0.001547, 0.001563, 0.001714, 0.001850, 0.001963, 0.002305, 0.002419, 0.0027, 0.0034, 0.00461, 0.006441, 0.007944, 0.006149, 0.006162]
	# desired_digits_2      = [0.8058, 0.8219, 0.8380, 0.8554, 0.8694, 0.9065, 0.9400, 0.9562, 0.9643, 0.9813, 0.9930, 0.9978, 0.9985, 0.9986]
	# wrong_digits_2        = [0.001537, 0.001556, 0.00178, 0.001875, 0.002059, 0.002289, 0.002617, 0.002678, 0.003750, 0.005110, 0.0065, 0.006017, 0.006140, 0.006188]
	# # probabilities without context
	# ddop_without_context = [0.8061, 0.8212, 0.8374, 0.8531, 0.8683, 0.9024, 0.9227, 0.9419, 0.95, 0.9487, 0.9572, 0.9952, 0.996, 0.9624]
	# wdop_without_context = [0.004874, 0.00497, 0.005084, 0.005271, 0.005818, 0.006361, 0.007406, 0.007386, 0.007832, 0.01377, 0.02310, 0.02949, 0.03193, 0.04566]
	# ddop_2               = [0.806, 0.820, 0.8382, 0.8525, 0.8680, 0.9037, 0.9302, 0.9424, 0.9471, 0.9649, 0.9942, 0.9800, 0.9804, 0.9987]
	# wdop_2               = [0.004912, 0.004900, 0.005044, 0.005086, 0.005581, 0.00602, 0.006634, 0.007012, 0.009401, 0.01588, 0.02521, 0.02671, 0.03178, 0.04127]
	# ddop_3               = [0.8064, 0.8218, 0.8385, 0.8518, 0.8677, 0.9058, 0.9335, 0.9421, 0.9520, 0.9645, 0.971, 0.998, 0.9914, 0.9806]
	# wdop_3               = [0.004783, 0.004848, 0.005047, 0.005257, 0.005706, 0.006264, 0.006273, 0.007239, 0.008189, 0.01592, 0.02375, 0.02503, 0.03330, 0.04343]
	# #without context
	# fig,ax=plt.subplots(2,1,sharex=True)
	# # with context (black)
	# ax[0].plot(p,desired_digits_over_p,"-^",color="k",label="With Context")
	# ax[1].plot(p,wrong_digits_over_p,"-^",color="k")
	# ax[0].plot(p,desired_digits_2,"-^",color="k")
	# ax[1].plot(p,wrong_digits_2,"-^",color="k")
	# # without context (red)
	# ax[0].plot(p,ddop_without_context,"-o",color="r",label="Without Context")
	# ax[1].plot(p,wdop_without_context,"-o",color="r")
	# ax[0].plot(p,ddop_2,"-o",color="r")
	# ax[1].plot(p,wdop_2,"-o",color="r")
	# ax[0].plot(p,ddop_3,"-o",color="r")
	# ax[1].plot(p,wdop_3,"-o",color="r")

	# ax[0].legend(loc="best")
	# ax[0].set_ylabel("Probability")
	# ax[0].set_title("Correct Digit")
	# ax[1].set_title("Wrong Digits")
	# ax[1].set_ylabel("Probability")
	# ax[1].set_xlabel("Multiplicator p")

	################################################################
	# testing the same but with modification as bias as david suggested, with 100 samples , 15x15 h1 units and 100 digits inputs
	if 0:
		p  = [0.5,1,1.5,2,5,7.5,10]

		# dd= desired digit, wd= wrong digit , c=context, nc=no context 
		# h2 as bias
		dd_with_context    = [[0.845553 , 0.8711,0.8899,0.908,0.977,0.99,0.991],[0.8452, 0.86989999, 0.889, 0.90700001, 0.97680002, 0.99070001, 0.9914],[0.84670001, 0.86970001, 0.89060003, 0.90670002, 0.97799999, 0.99070001, 0.99199998]]
		dd_without_context = [[0.849274 ,0.868754 ,0.887845 ,0.898609,0.906878 , 0.879146 ,0.852557 ],[0.8477, 0.86940002, 0.88679999, 0.90020001, 0.90609998, 0.88050002, 0.85259998],[0.847, 0.86970001, 0.88669997, 0.89990002, 0.90609998, 0.87900001, 0.8527]]
		wd_with_context    = [[0.00264,0.00307,0.0036,0.0045,0.028,0.06,0.11],[0.0026, 0.0029, 0.0035000001, 0.0044999998, 0.027799999, 0.069499999, 0.1098],[0.0026, 0.0031000001, 0.0037, 0.0044999998, 0.028000001, 0.069300003, 0.11]]
		wd_without_context = [[0.00379,0.006,0.009,0.012,0.088,0.18,0.24],[0.0038999999, 0.0060000001, 0.0088999998, 0.013, 0.088399999, 0.1788, 0.2482],[0.0038000001, 0.0060000001, 0.0087000001, 0.0128, 0.088399999, 0.1787, 0.2484]]

		# h2 clamped with multiplcation
		dd_context  = [[0.8367, 0.85079, 0.86800, 0.88475, 0.94230, 0.96696, 0.97930],[0.83740001, 0.85180002, 0.8681999, 0.88235002, 0.94233999, 0.95167999, 0.9691499],[0.83840000, 0.85430002, 0.8688000, 0.88270002, 0.94247999, 0.95963999, 0.96673002]]
		dd_ncontext = [[0.83780, 0.85229, 0.86686, 0.8805, 0.93409, 0.95723, 0.96122],[0.8375999, 0.8529999, 0.86633332, 0.88164997, 0.9348199, 0.95102666, 0.96147003],[0.8367999, 0.85049998, 0.86519996, 0.87964999, 0.93521995, 0.95128002, 0.97672996]]
		wd_context  = [[0.00060000, 0.00060000, 0.00066666, 0.000699, 0.001119, 0.0027866, 0.0041000],[0.00060000, 0.000699, 0.00066666, 0.00065000, 0.0011599, 0.0021733, 0.003099],[0.00060000, 0.00060000, 0.00066666, 0.00065000, 0.0012000, 0.00294, 0.00406]]
		wd_ncontext = [[0.0024000, 0.0026000, 0.0028666, 0.0031999, 0.0058400, 0.0092400, 0.013150],[0.0024000, 0.0027000, 0.0028666, 0.0032500, 0.0060199, 0.0082399, 0.011140],[0.0026000, 0.0027000, 0.0027333, 0.0034000, 0.0064599, 0.010186, 0.01094]]

		# plot
		fig,ax=plt.subplots(2,1,sharex=True)

		seaborn.tsplot(dd_with_context,p,color="red",ax=ax[0],err_style="ci_bars")
		seaborn.tsplot(dd_without_context,p,linestyle="--",color="red",ax=ax[0],err_style="ci_bars")
		seaborn.tsplot(wd_with_context,p,color="red",condition="as bias / with context",ax=ax[1],err_style="ci_bars")
		seaborn.tsplot(wd_without_context,p,linestyle="--",color="red",condition="as bias / without context",ax=ax[1],err_style="ci_bars")


		seaborn.tsplot(dd_context,p,color="blue",ax=ax[0],err_style="ci_bars")
		seaborn.tsplot(dd_ncontext,p,linestyle="--",color="blue",ax=ax[0],err_style="ci_bars")
		seaborn.tsplot(wd_context,p,color="blue",condition="clamped / with context",ax=ax[1],err_style="ci_bars")
		seaborn.tsplot(wd_ncontext,p,linestyle="--",color="blue",condition="clamped / without context",ax=ax[1],err_style="ci_bars")

		# plt.legend(loc="best")
		plt.xlabel("Multiplicator n")
		ax[0].set_ylabel("Probability")
		ax[1].set_ylabel("Probability")
		ax[1].set_xticks(range(11))
		ax[0].set_title("Desired Digit")
		ax[1].set_title("Wrong Digits")


plt.show()
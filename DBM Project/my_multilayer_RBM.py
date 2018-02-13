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

	mpl.rcParams["image.cmap"] = "jet"
	mpl.rcParams["grid.linewidth"] = 0.5
	mpl.rcParams["lines.linewidth"] = 1.
	mpl.rcParams["font.family"]= "serif"
	# plt.rcParams['image.cmap'] = 'coolwarm'
	# seaborn.set_palette(seaborn.color_palette("Set2", 10))

	log=Logger(True)




from tensorflow.examples.tutorials.mnist import input_data
time_now = time.asctime()

#### Load MNIST Data 
if "train_data" not in globals():
	log.out("Loading Data")
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	train_data, train_label, test_data, test_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
	test_data[:]=test_data[:]>0.3
	train_data[:]=train_data[:]>0.3
	#get test data of only one number class:
	index_for_number=[]
	for i in range(len(test_label)):
		if (test_label[i]==[0,0,0,1,0,0,0,0,0,0]).sum()==10:
			index_for_number.append(i)

	half_images = test_data[0:11]
	#halfing some images from test_data
	half_images[1:6,500:] = 0

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
		""" shape definitions are wired here-normally one would define [rows,columns] but here it is reversed with [columns,rows]? """

		self.v       = tf.placeholder(tf.float32,[None,self.visible_units],name="Visible-Layer") # has shape [number of images per batch,number of visible units]

		self.w       = tf.Variable(tf.random_uniform([self.visible_units,self.hidden_units],minval=-1e-3,maxval=1e-3),name="Weights")# init with small random values to break symmetriy
		self.bias_v  = tf.Variable(tf.zeros([self.visible_units]),name="Visible-Bias")
		self.bias_h  = tf.Variable(tf.zeros([self.hidden_units]),name="Hidden-Bias")


		# get the probabilities of the hidden units in 
		self.h_prob  = sigmoid(tf.matmul(self.v,self.forw_mult*self.w) + self.bias_h,temp)
		#h has shape [number of images per batch, number of hidden units]
		# get the actual activations for h {0,1}
		self.h       = tf.nn.relu(
			            tf.sign(
			            	self.h_prob - tf.random_uniform(tf.shape(self.h_prob)) 
			            	) 
		        		) 

		# and the same for visible units
		self.v_prob  = sigmoid(tf.matmul(self.h,tf.transpose(self.back_mult*self.w)) + self.bias_v,temp)
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
		self.pos_grad  = tf.matmul(tf.transpose(self.v),self.h)
		self.neg_grad  = tf.matmul(tf.transpose(self.v_recon),self.h_gibbs)
		self.numpoints = tf.cast(tf.shape(self.v)[0],tf.float32) #number of train inputs per batch (for averaging the CD matrix -> see practical paper by hinton)
		# contrastive divergence
		self.CD        = (self.pos_grad - self.neg_grad)/self.numpoints
		

		#update w
		self.update_w = self.w.assign(self.w+self.learnrate*self.CD)
		self.mean_w   = tf.reduce_mean(self.w)
		#update bias
		""" Since vectors v and h are actualy matrices with number of batch_size images in them, reduce mean will make them to a vector again """
		self.update_bias_v = self.bias_v.assign(self.bias_v+self.learnrate*tf.reduce_mean(self.v-self.v_recon,0))
		self.update_bias_h = self.bias_h.assign(self.bias_h+self.learnrate*tf.reduce_mean(self.h-self.h_gibbs,0))


		# reverse feed
		self.h_rev       = tf.placeholder(tf.float32,[None,self.hidden_units],name="Reverse-hidden")
		self.v_prob_rev  = sigmoid(tf.matmul(self.h_rev,tf.transpose(self.w)) + self.bias_v,temp)
		self.v_recon_rev = tf.nn.relu(tf.sign(self.v_prob_rev - tf.random_uniform(tf.shape(self.v_prob_rev))))

	def train(self,sess,RBM_i,RBMs,batch):
		self.my_input_data = batch
		if RBM_i==1:
			self.my_input_data=RBMs[RBM_i-1].h.eval({RBMs[RBM_i-1].v:batch})
		elif RBM_i==2:
			self.my_input_data=RBMs[RBM_i-1].h.eval({RBMs[RBM_i-1].v:RBMs[RBM_i-1].my_input_data})

		#### update the weights and biases
		self.w_i,self.error_i=sess.run([self.update_w,self.error],feed_dict={self.v:self.my_input_data})
		sess.run([self.update_bias_h,self.update_bias_v],feed_dict={self.v:self.my_input_data})

		return self.w_i,self.error_i




################################################################################################################################################
### Class Deep BM 
class DBM_class(object):
	"""defines a deep boltzmann machine
	"""

	def __init__(self,shape,learnrate,liveplot):
		self.n_layers     = len(shape)
		self.liveplot     = liveplot
		self.shape        = shape  # contains the number of  neurons in a list from v layer to h1 to h2 
		
		
		self.learnrate    = learnrate
		
		self.init_state     = 0
		self.exported       = 0
		self.m              = 0 #laufvariable
		self.num_of_skipped = 50 # how many tf.array value adds get skipped 
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
		self.RBMs[0] = RBM(self.shape[0],self.shape[1], 2, 1, learnrate=rbm_learnrate, liveplot=0)
		self.RBMs[1] = RBM(self.shape[1],self.shape[2], 1, 2, learnrate=rbm_learnrate, liveplot=0)


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
				for RBM_i,RBM in enumerate(self.RBMs):
					log.start("Pretraining ",str(RBM_i+1)+".", "RBM")

					for epoch in range(pretrain_epochs):
						log.start("Epoch:",epoch+1,"/",pretrain_epochs)
						
						for start, end in zip( range(0, len(train_data), batchsize_pretrain), range(batchsize_pretrain, len(train_data), batchsize_pretrain)):
							#### define a batch
							batch = train_data[start:end]
							# train the rbm 
							w_i,error_i = RBM.train(sess,RBM_i,self.RBMs,batch)
							#### liveplot
							if RBM.liveplot and plt.fignum_exists(fig.number) and start%40==0:
								ax.cla()
								rbm_shape=int(sqrt(RBM.visible_units))
								matrix_new=tile_raster_images(X=w_i.T, img_shape=(rbm_shape, rbm_shape), tile_shape=(10, 10), tile_spacing=(0,0))
								ax.matshow(matrix_new)
								plt.pause(0.00001)


						log.info("Learnrate:",round(rbm_learnrate,4))
						log.info("error",round(error_i,4))
						log.end() #ending the epoch


					log.end() #ending training the rbm 

				

				# define the weights
				self.weights=[]
				for i in range(len(self.RBMs)):
					self.weights.append(self.RBMs[i].w.eval())

				if save_pretrained:
					for i in range(len(self.weights)):
						np.savetxt("Pretrained-"+" %i "%i+str(time_now)+".txt", self.weights[i])
						log.out("Saved Pretrained")
			else:
				self.weights=[]
				log.out("Loading Pretrained from file")
				for i in range(len(self.shape)-1):
					self.weights.append(np.loadtxt("Pretrained-"+" %i "%i+pathsuffix_pretrained+".txt").astype(np.float32))

			log.end()


	def load_from_file(self,path):
		os.chdir(path)
		log.out("Loading data from:","...",path[-20:])
		self.w1_np=np.loadtxt("w1.txt")
		self.w2_np=np.loadtxt("w2.txt")
		self.bias1_np=np.loadtxt("bias1.txt")
		self.bias2_np=np.loadtxt("bias2.txt")
		self.bias3_np=np.loadtxt("bias3.txt")
		os.chdir(workdir)

	
	def import_(self):
		""" setting up the graph and setting the weights and biases tf variables to the 
		saved numpy arrays """
		log.out("loading numpy vars into graph")
		sess.run(self.w1.assign(self.w1_np))
		sess.run(self.w2.assign(self.w2_np))
		sess.run(self.bias_v.assign(self.bias1_np))
		sess.run(self.bias_h1.assign(self.bias2_np))
		sess.run(self.bias_h2.assign(self.bias3_np))


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

		# these variables need to be init to random for use in h1 , later they get changed to something else
		self.h2 = tf.Variable(tf.random_uniform([self.batchsize,self.shape[2]],minval=-1e-3,maxval=1e-3),name="h2")
		# self.v_rev   = tf.Variable(tf.random_uniform([self.batchsize,self.shape[0]],minval=-1e-3,maxval=1e-3),name="v_rev_init")

		tf.variables_initializer([self.h2], name='init_train')


		self.v  = tf.placeholder(tf.float32,[None,self.shape[0]],name="Visible-Layer") # has self.shape [number of images per batch,number of visible units]
		
		if graph_mode=="gibbs":
			self.temp=tf.placeholder(tf.float32,[],name="Temperature")
		else:
			self.temp=temp

		
		if graph_mode=="training":
			self.m_tf = tf.placeholder(tf.int32,[],name="running_array_index")
			self.h2   = tf.placeholder(tf.float32,[None,self.shape[2]],name="placeholder_h2")

			self.h1_activity_ = tf.Variable(tf.zeros([self.num_of_updates/self.num_of_skipped]))
			self.h2_activity_ = tf.Variable(tf.zeros([self.num_of_updates/self.num_of_skipped]))
			self.train_error_ = tf.Variable(tf.zeros([self.num_of_updates/self.num_of_skipped]))
			self.w1_mean_     = tf.Variable(tf.zeros([self.num_of_updates/self.num_of_skipped]))
			#self.CD1_mean_    = tf.Variable(tf.zeros([self.num_of_updates/self.num_of_skipped]))
			#self.CD2_mean_    = tf.Variable(tf.zeros([self.num_of_updates/self.num_of_skipped]))		


		# self.h2      = tf.placeholder(tf.random_uniform([None,self.shape[1].hidden_units],minval=-1e-3,maxval=1e-3),name="h2_placeholder")

		self.w1 = tf.Variable(self.weights[0],name="Weights1")
		self.w2 = tf.Variable(self.weights[1],name="Weights2")

		self.bias_v  = tf.Variable(tf.zeros([self.shape[0]]),name="Visible-Bias")
		self.bias_h1 = tf.Variable(tf.zeros([self.shape[1]]), name="Hidden-Bias")
		self.bias_h2 = tf.Variable(tf.zeros([self.shape[2]]), name="Hidden-Bias")

		# self.temp_tf     = tf.Variable(temp, name="Temperature")
	
		### Propagation
		## Forward Feed
		# h1 gets both inputs from h2 and v
		self.h1_prob = sigmoid(tf.matmul(self.v,self.w1) + tf.matmul(self.h2,self.w2,transpose_b=True) + self.bias_h1,self.temp)
		self.h1      = self.sample(self.h1_prob)
		
		# h2 only from h1
		if graph_mode=="testing" or graph_mode=="gibbs":
			self.h2_prob = sigmoid(tf.matmul(self.h1,self.w2) + self.bias_h2,self.temp)
			self.h2      = self.sample(self.h2_prob)

		## Backward Feed   
		self.h1_recon_prob = sigmoid(tf.matmul(self.v,self.w1)+tf.matmul(self.h2,self.w2,transpose_b=True)+self.bias_h1, self.temp)
		self.h1_recon      = self.sample(self.h1_recon_prob)
		self.v_recon_prob  = sigmoid(tf.matmul(self.h1,self.w1,transpose_b=True)+self.bias_v, self.temp)
		self.v_recon       = self.sample(self.v_recon_prob)

		## Gibbs step 
		self.h1_gibbs_prob = sigmoid(tf.matmul(self.v_recon_prob,self.w1) + tf.matmul(self.h2,self.w2,transpose_b=True) + self.bias_h1,self.temp)
		self.h1_gibbs      = self.sample(self.h1_gibbs_prob)
		self.h2_gibbs_prob = sigmoid(tf.matmul(self.h1_recon_prob,self.w2), self.temp)
		self.h2_gibbs      = self.sample(self.h2_gibbs_prob)

		
		## Backward Feed 2
		self.h1_recon_prob2 = sigmoid(tf.matmul(self.v_recon,self.w1)+tf.matmul(self.h2_gibbs,self.w2,transpose_b=True)+self.bias_h1, self.temp)
		self.h1_recon2      = self.sample(self.h1_recon_prob2)
		self.v_recon_prob2  = sigmoid(tf.matmul(self.h1_recon2,self.w1,transpose_b=True)+self.bias_v, self.temp)
		self.v_recon2       = self.sample(self.v_recon_prob2)

		## Gibbs step 2
		self.h1_gibbs_prob2 = sigmoid(tf.matmul(self.v_recon_prob2,self.w1) + tf.matmul(self.h2_gibbs,self.w2,transpose_b=True) + self.bias_h1,self.temp)
		self.h1_gibbs2      = self.sample(self.h1_gibbs_prob2)
		self.h2_gibbs_prob2 = sigmoid(tf.matmul(self.h1_recon_prob2,self.w2), self.temp)
		self.h2_gibbs2      = self.sample(self.h2_gibbs_prob2)

		
		#Error and stuff
		self.error  = tf.reduce_mean(tf.square(self.v-self.v_recon))
		self.h1_sum = tf.reduce_sum(self.h1)
		self.h2_sum = tf.reduce_sum(self.h2)

		### Training with contrastive Divergence
		#first weight matrix
		self.pos_grad1  = tf.matmul(self.v, self.h1_prob,transpose_a=True)
		self.neg_grad1  = tf.matmul(self.v_recon,self.h1_gibbs,transpose_a=True)
		self.numpoints1 = tf.cast(tf.shape(self.v)[0],tf.float32) #number of train inputs per batch (for averaging the CD matrix -> see practical paper by hinton)
		# self.weight_decay1 = tf.reduce_sum(tf.abs(self.w1))*0.00001
		self.CD1        = (self.pos_grad1 - self.neg_grad1)/self.numpoints1
		# self.CD1_mean   = tf.reduce_mean(tf.square(self.CD1))
		self.update_w1  = self.w1.assign(self.w1+self.learnrate*self.CD1)
		self.mean_w1    = tf.reduce_mean(tf.square(self.w1))
		# second weight matrix
		self.pos_grad2  = tf.matmul(self.h1, self.h2,transpose_a=True)
		self.neg_grad2  = tf.matmul(self.h1_recon, self.h2_gibbs,transpose_a=True)
		self.numpoints2 = tf.cast(tf.shape(self.h2)[0],tf.float32) #number of train inputs per batch (for averaging the CD matrix -> see practical paper by hinton)
		self.CD2	    = (self.pos_grad2 - self.neg_grad2)/self.numpoints2
		# self.CD2_mean   = tf.reduce_mean(tf.square(self.CD2))
		self.update_w2  = self.w2.assign(self.w2+self.learnrate*self.CD2)
		# bias updates
		self.update_bias_h1 = self.bias_h1.assign(self.bias_h1+self.learnrate*tf.reduce_mean(self.h1-self.h1_gibbs,0))
		self.update_bias_h2 = self.bias_h2.assign(self.bias_h2+self.learnrate*tf.reduce_mean(self.h2-self.h2_gibbs,0))
		self.update_bias_v  = self.bias_v.assign(self.bias_v+self.learnrate*tf.reduce_mean(self.v-self.v_recon,0))
		
		
		if graph_mode=="training":
			self.assign_arrays =	[tf.scatter_update(self.train_error_,self.m_tf,self.error),
							 #tf.scatter_update(self.CD1_mean_,self.m_tf,self.CD1_mean),
							 #tf.scatter_update(self.CD2_mean_,self.m_tf,self.CD2_mean),
							 tf.scatter_update(self.w1_mean_,self.m_tf,self.mean_w1),
							 tf.scatter_update(self.h1_activity_,self.m_tf,self.h1_sum),
							 tf.scatter_update(self.h2_activity_,self.m_tf,self.h2_sum),
							]

		### reverse feed
		self.h2_rev      = tf.placeholder(tf.float32,[None,10],name="reverse_h2")
		self.h1_rev_prob = sigmoid(tf.matmul(self.v, self.w1)+tf.matmul(self.h2_rev, (self.w2),transpose_b=True)+self.bias_h1,self.temp)
		self.h1_rev      = tf.nn.relu(tf.sign(self.h1_rev_prob - tf.random_uniform(tf.shape(self.h1_rev_prob)))) 
		self.v_rev_prob  = sigmoid(tf.matmul(self.h1_rev, (self.w1),transpose_b=True)+self.bias_v,self.temp)
		self.v_rev       = tf.nn.relu(tf.sign(self.v_rev_prob - tf.random_uniform(tf.shape(self.v_rev_prob)))) 

		#test sample
		self.h1_place=tf.placeholder(tf.float32,[None,self.shape[1]],name="h1_placeholder")
		self.h2_sample=sigmoid(tf.matmul(self.h1_place,self.w2) + self.bias_h2, self.temp)

		sess.run(tf.global_variables_initializer())
		self.init_state=1

	def test_noise_stability(self,input_data,input_label):
		self.batchsize=len(input_data)
		if load_from_file:
			self.load_from_file(workdir+"/data/"+pathsuffix)
		self.graph_init("testing")
		self.import_()

		n=20
		h2_=[]
		r=rnd.random([self.batchsize,784])
		v_noise=np.copy(input_data)
		
		for i in range(200):
			h2            = self.h2_prob.eval({self.v:v_noise})
			v_noise_recon = self.v_recon_prob.eval({self.v:v_noise})
			
			for i in range(n):
				v_noise_recon+=self.v_recon_prob.eval({self.v:v_noise})
			v_noise_recon*=1./(n+1)
			
			# classify the reconstructed image
			h1 = self.h1.eval({self.v:v_noise})
			for i in range(n):
				h1 += self.h1_rev.eval({self.v:v_noise_recon,self.h2_rev:[[1,1,-3,1,-3,1,1,1,1,1]]})
			h1*=1./(n+1)
			for i in range(n):	
				h2 += self.h2_sample.eval({self.h1_place:h1})
			h2*=1./(n+1)
			
			
			# make the input more noisy
			v_noise += (abs(r-0.5)*0.01)
			v_noise*=1./v_noise.max()

			h2_.append(h2[0])
		
		return np.array(h2_),v_noise_recon

	def train(self,epochs,num_batches,cont):
		""" training the DBM with given h2 as labels """
		# init all vars for training
		self.batchsize      = int(55000/num_batches)
		num_of_updates      = epochs*num_batches
		self.num_of_updates = num_of_updates
		d_learnrate         = float(dbm_learnrate_end-self.learnrate)/num_of_updates
		self.m              = 0

		
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
			fig,ax=plt.subplots(1,1,figsize=(15,10))

		
		# starting the training
		for epoch in range(epochs):
			log.start("Deep BM Epoch:",epoch+1,"/",epochs)

			for start, end in zip( range(0, len(train_data), self.batchsize), range(self.batchsize, len(train_data), self.batchsize)):
				
				# define a batch
				batch = train_data[start:end]
				batch_label = train_label[start:end]

				# run all updates 
				sess.run([	self.update_w1,
						self.update_w2,
						self.update_bias_v,
						self.update_bias_h1,
						self.update_bias_h2,
						],
						feed_dict={	self.v:batch,
								self.h2:batch_label}
					)
				
				# add values to the tf.arrays
				if self.m%self.num_of_skipped==0:
					try:
						sess.run([self.assign_arrays],
								feed_dict={	self.v:batch,
										self.h2:batch_label,
										self.m_tf:self.m/self.num_of_skipped}
							)
					except:
						log.info("Error: Entry in tf arrays not possible for m="+str(self.m))
				# increase the learnrate
				self.learnrate+=d_learnrate

				self.m+=1

				
				if self.liveplot and plt.fignum_exists(fig.number) and start%40==0:
					ax.cla()
					matrix_new=tile(self.w1.eval())
					ax.matshow(matrix_new)
					plt.pause(0.00001)



			# self.train_error_np=self.train_error_.eval()
			# log.out("error:",np.round(self.train_error_np[m],4)," learnrate:",self.learnrate)
			
			log.end() #ending the epoch
		log.reset()

		# normalize the activity arrays
		self.h1_activity_*=1./(n_second_layer*self.batchsize)

		self.export()


	def test(self,test_data):
		""" testing runs without giving h2 , only v is given and h2 has to be infered 
		by the DBM """
		#init the vars and reset the weights and biases 
		
		self.batchsize=1

		# "verarsche" tf graph init weil h2 einfach nur fur die erste def gebraucht wird im graph und danach anders definiert
		# self.h2   = tf.Variable(tf.random_uniform([len(test_data),self.shape[2]],minval=-1e-3,maxval=1e-3),name="h2")
		# self.v_rev   = tf.Variable(tf.random_uniform([len(test_data),self.shape[0]],minval=-1e-3,maxval=1e-3),name="v_rev_init")
		# tf.variables_initializer([self.h2,self.v_rev], name='init_train')

		if load_from_file and not training:
			self.load_from_file(workdir+"/data/"+pathsuffix)

		self.graph_init("testing") # "testing" because this graph creates the testing variables where only v is given, not h2

		self.import_()

		log.start("Testing DBM")
		self.test_error  = self.error.eval({self.v:test_data})
		self.h1_act_test = self.h1_sum.eval({self.v:test_data})
		self.h2_act_test = self.h2_sum.eval({self.v:test_data})

		self.probs      = self.v_recon_prob.eval({self.v:test_data})
		self.rec        = self.v_recon.eval({self.v:test_data})

		# self.rec_h1  = self.h1_recon_prob.eval({self.v:test_data})
		# self.h1_test = self.h1_prob.eval({self.v:test_data})
		self.h2_test   = self.h2_prob.eval({self.v:test_data})

		N=20
		log.info("sampling h2 with %i steps"%N)
		for i in range(N):
			h1           = self.h1_rev.eval({self.v:test_data,self.h2_rev:self.h2_test})
			self.h2_test = self.h2_sample.eval({self.h1_place:h1})


		log.end()


		self.h1_act_test*=1./(n_second_layer*len(test_data))
		self.h2_act_test*=1./(n_third_layer*len(test_data))

		self.test_error_.append(self.test_error)
		# error of classifivation labels
		self.class_error=np.mean(np.abs(self.h2_test-test_label))
		
		# #set the maximum = 1 and the rest 0 		
		# log.out("Taking only the maximum")
		# for i in range(10000):
		# 	self.h2_test[i]=self.h2_test[i]==self.h2_test[i].max()

		log.reset()
		log.info("Reconstr. error: ",np.round(DBM.test_error,5), "learnrate: ",np.round(dbm_learnrate,5))
		log.info("Class error: ",np.round(self.class_error,5))
		log.info("Activations of Neurons: ", np.round(self.h1_act_test,4) , np.round(self.h2_act_test,4))




	def gibbs_sampling(self,v_input,gibbs_steps,temp_start,temp_end,modification,mode,p=1,liveplot=1):
		""" Repeatedly samples v and h2 , where h2 can be modified by the user with the multiplication
		by the modification array - clamping the labels to certain numbers.
		v_input :: starting with an image as input 
		
		temp_end, temp_start :: temperature will decrease or increase to temp_end and start at temp_start 
		
		mode 	:: "sampling" calculates h2 and v back and forth usign previous steps
			:: "context" clamps v and only calculates h1 based on previous h2
		
		p :: multiplication of the h2 array to increase the importance of the layer
		"""

		# self.v_rev = tf.Variable(tf.random_uniform([1,self.shape[0]],minval=-1e-3,maxval=1e-3),name="v_rev_init")
		# self.h2    = tf.Variable(tf.random_uniform([1,self.shape[2]],minval=-1e-3,maxval=1e-3),name="h2")
		
		# tf.variables_initializer([self.h2,self.v_rev], name='init_train')

		self.batchsize = 1
		h2_            = np.zeros([gibbs_steps,self.shape[2]])
		h1_            = np.zeros([gibbs_steps,self.shape[1]])
		v_g_           = np.zeros([gibbs_steps,self.shape[0]])
		temp_          = np.zeros([gibbs_steps])
		temp           = temp_start
		temp_delta     = (temp_end-temp_start)/gibbs_steps

		self.num_of_updates=1000 #just needs to be defined because it will make a train graph with tf.arrays where this number is needed


		# tf.variables_initializer([temp], name='init_train')

		if liveplot:
			log.info("Liveplotting gibbs sampling")
			fig,ax=plt.subplots(1,3,figsize=(12,5))
		
		# set v as input 
		v_gibbs = v_input 
		
		#calculate forward feed h2
		h2 = self.h2_prob.eval({self.v:v_gibbs,self.temp:1.0})
		h1 = self.h1_rev.eval({self.v:v_input,self.h2_rev:h2,self.temp:temp})


		for i in range(gibbs_steps):
			# if liveplot and not plt.fignum_exists(fig.number):
			# 	break 

			if mode=="sampling":
				# calculate the backward and forward pass 
				v_gibbs = self.v_rev_prob.eval({self.v:v_gibbs, self.h2_rev:p*h2, self.temp:temp})
				h2      = self.h2_prob.eval({self.v:v_gibbs, self.temp:temp})
				
				# here the h2 vector can be changed like: h2[0][1:4]=0
				h2[0]*=modification

			
				
			if mode=="context":
				# v is clamped here , calc only h1 and h2 , v_gibbs only for plotting and will not be used to calc h1 or h2
				if liveplot:
					v_gibbs = self.v_rev_prob.eval({self.v:	 v_gibbs, 
								  	   	  self.h2_rev: h2, 
									   	  self.temp:	 temp})
				
				h1 = self.h1_rev.eval({	self.v:	 v_input,
								self.h2_rev: p*np.reshape(modification,[1,10]),
								self.temp:	 temp})

				h2 = self.h2_sample.eval({self.h1_place: 	h1,
									self.temp: 	temp})
				
				# here the h2 vector can be changed like: h2[0][1:4]=0
				# h2[0]*=modification
			
			# assign new temp
			temp+=temp_delta 

			# save values to array
			h2_[i]   = h2[0]
			h1_[i]   = h1
			v_g_[i]  = v_gibbs
			temp_[i] = temp


		if liveplot and plt.fignum_exists(fig.number):
			a0=ax[0].matshow(v_g_[0].reshape(28,28))
			if mode=="context":
				a2=ax[2].matshow(h1_[0].reshape(int(sqrt(self.shape[1])),int(sqrt(self.shape[1]))))
			a1,=ax[1].plot(h2_[0],"-o")
			ax[1].set_ylim(0,1)
			ax[1].grid()
			ax[1].set_title("Temp.: %s, Steps: %s"%(str(round(temp_[0],3)),str(i)))
			for i in range(1,len(h2_)):
				ax[1].set_title("Temp.: %s, Steps: %s"%(str(round(temp_[i],3)),str(i)))
				
				a0.set_data(v_g_[i].reshape(28,28))
				a1.set_data(range(10),h2_[i])
				a2.set_data(h1_[i].reshape(int(sqrt(self.shape[1])),int(sqrt(self.shape[1]))))
				
				plt.pause(1/30.)
		
			plt.close(fig)


		return v_gibbs,np.array(h2_)

	def export(self):
		# convert weights and biases to numpy arrays
		self.w1_np    = self.w1.eval() 
		self.w2_np    = self.w2.eval()
		self.bias1_np = self.bias_v.eval()
		self.bias2_np = self.bias_h1.eval()
		self.bias3_np = self.bias_h2.eval()

		# convert tf.arrays to numpy arrays 
		if training:
			self.h1_activity_np = self.h1_activity_.eval()
			self.h2_activity_np = self.h2_activity_.eval()
			self.train_error_np = self.train_error_.eval()
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
		np.savetxt("bias1.txt", self.bias1_np)
		np.savetxt("bias2.txt", self.bias2_np)
		np.savetxt("bias3.txt", self.bias3_np)
		
		self.log_list.append(["train_time",self.train_time])

		with open("logfile.txt","w") as log_file:
				for i in range(len(self.log_list)):
					log_file.write(self.log_list[i][0]+","+str(self.log_list[i][1])+"\n")
		
		log.info("Saved data and log to:",new_path)


		if save_all_params:
			if training:
				np.savetxt("h1_activity.txt", self.h1_activity_np)
				np.savetxt("train_error.txt", self.train_error_np)
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

num_batches_pretrain = 500
dbm_batches          = 500
pretrain_epochs      = 1
dbm_epochs           = 50


rbm_learnrate     = 0.005
dbm_learnrate     = 0.01
dbm_learnrate_end = 0.01  # bringt nichts

temp = 1.0		

pre_training    = 0 	#if no pretrain then files are automatically loaded

training        = 0
testing         = 0
plotting        = 0
gibbs_sampling  = 1
noise_stab_test = 0

save_to_file          = 0 	# only save biases and weights for further training
save_all_params       = 0	# also save all test data and reconstructed images (memory heavy)
save_pretrained       = 0

load_from_file        = 1
pathsuffix            = "Sun Feb 11 20-20-39 2018"#"Thu Jan 18 20-04-17 2018 80 epochen"
pathsuffix_pretrained = "Thu Jan 25 11-28-08 2018"


number_of_layers = 3
n_first_layer    = 784
n_second_layer   = 15*15
n_third_layer    = 10

saveto_path=data_dir+"/"+time_now

### modify the parameters with additional_args
if len(additional_args)>0:
	n_second_layer = (10+int(additional_args[0]))*(10+int(additional_args[0]))
	saveto_path    += " - "+str(additional_args[0])


######### DBM ##########################################################################
#### Pre training is ended - create the DBM with the gained weights
# if i == 0,1,2,...: (das ist das i von der echo cluster schleife) in der dbm class stehen dann die parameter fur das jeweilige i 
DBM = DBM_class(	shape=[n_first_layer,n_second_layer,n_third_layer],
			learnrate = dbm_learnrate,
			liveplot=0
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
			
			
			DBM.train(	epochs = dbm_epochs,
					num_batches = dbm_batches,
					cont=i)

			DBM.train_time=log.end()

	# test session
	if testing:
		with tf.Session() as sess:
			log.start("Test Session")
			# new session for test images - v has 10.000 length 
			#testing the network , this also inits the graph so do not comment it out
			DBM.test(test_data) 


			log.end()


if gibbs_sampling:
	with tf.Session() as sess:
		log.start("Gibbs Sampling Session")

		if load_from_file and not training:
			DBM.load_from_file(workdir+"/data/"+pathsuffix)

		DBM.batchsize=1
		DBM.graph_init("gibbs")
		DBM.import_()


		dd_c=[]
		dd_nc=[]
		wd_c=[]
		wd_nc=[]
		for p in [1]: 
			log.start("p =",p) 
			# arrays for h2 act without context
			desired_digits_c  = []
			wrong_digits_c    = []  
			# arrays for h2 act without context 
			desired_digits_nc = []
			wrong_digits_nc   = []

			check_wrongs=[]
			# multiplication for the whole h2 layer:
			# p = 1

			# loop through images from test_data
			for i in range(18,200):
				## find the digit that was presented
				digit=np.where(test_label[i])[0][0] 
				## set desired digit range
				if digit<5:
					# calculte h2 firerates over all gibbs_steps with no context
					_,h2_2_no_context=DBM.gibbs_sampling(test_data[i:i+1], 200, 1. , 0.2, 
											mode         = "context", 
											modification = np.array([1,1,1,1,1,1,1,1,1,1]),
											p            = p,
											liveplot     = 0)
					# with context
					_,h2_2_context=DBM.gibbs_sampling(test_data[i:i+1], 200, 1. , 0.2, 
											mode         = "context", 
											modification = np.array([1,1,1,1,1,0,0,0,0,0]),
											p            = p,
											liveplot     = 0)
					
					# append h2 activity to array, but only the unit that corresponst to the given digit picture
					desired_digits_c.append(h2_2_context[:,digit])
					desired_digits_nc.append(h2_2_no_context[:,digit])
					# append all other h2 activities 
					for i in range(10):
						if i!=digit:
							wrong_digits_c.append(h2_2_context[:,i])
							wrong_digits_nc.append(h2_2_no_context[:,i])

					# check if h2 has classified digits over 4
					check_wrongs.append(h2_2_no_context[-1][5:].max())
	
			log.out("With Context:")
			log.info("Desired Digits:\t",np.mean(desired_digits_c))
			log.info("Wrong Digits:\t",np.mean(wrong_digits_c))
			log.out("Without Context")
			log.info("Desired Digits:\t",np.mean(desired_digits_nc))
			log.info("Wrong Digits:\t",np.mean(wrong_digits_nc))

			dd_c.append(np.round(np.mean(desired_digits_c),4))
			dd_nc.append(np.round(np.mean(desired_digits_nc),4))
			wd_c.append(np.round(np.mean(wrong_digits_c),4))
			wd_nc.append(np.round(np.mean(wrong_digits_nc),4))
			# clac how many digits got badly classified
			wrong_class_nc=[np.sum(np.array(desired_digits_nc)[:,-1]<i) for i in np.linspace(0,1,100)]
			wrong_class_c=[np.sum(np.array(desired_digits_c)[:,-1]<i) for i in np.linspace(0,1,100)]

			log.end()

		### plot
		# fig_gs,ax_gs = plt.subplots(2,1,sharex="all")
		# for i in range(len(desired_digits_c)):
		# 	ax_gs[0].plot(smooth(desired_digits_c[i],20))
		# 	ax_gs[1].plot(smooth(desired_digits_nc[i],20))
		# 	ax_gs[0].set_title("Desired Digit with Context")
		# 	ax_gs[1].set_title("Desired Digit without Context")
		# 	ax_gs[0].set_ylim(0,1)
		# 	ax_gs[1].set_ylim(0,1)
		# 	plt.xlabel("gibbs_steps")
		# 	ax_gs[0].set_ylabel("Probability")
		# 	ax_gs[1].set_ylabel("Probability")

		plt.figure()
		plt.plot(np.linspace(0,1,100),wrong_class_c,"-",label="With Context")
		plt.plot(np.linspace(0,1,100),wrong_class_nc,"-",label="Without Context")
		plt.title("How many digits got classified below Threshhold")
		plt.xlabel("Threshhold")
		plt.ylabel("Number of Digits")
		plt.legend()


		# v_gibbs2,h2_2=DBM.gibbs_sampling(test_data[1:2], 500, modification=[1,1,1,0,1,1,1,1,1,1], liveplot=0)
		# plt.plot(smooth(h2_2[:,3],20))
		# print "Mean: 1: %f, 2: %f"%(np.mean(h2_1[:,3]),np.mean(h2_2[:,3]))

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
if plotting:
	log.out("Plotting...")
	map1=plt.matshow(tile(DBM.w1_np))
	plt.colorbar(map1)
	plt.title("W 1")

	# plt.matshow(tile(DBM.CD1_np))
	map2=plt.matshow(tile_raster_images(X=DBM.w2_np.T, img_shape=(15,15), tile_shape=(12, 12), tile_spacing=(0,0)))
	# plt.title("W 2")
	# plt.colorbar(map2)	

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
		ax_fr3.plot(x,DBM.train_error_np,"k")
		ax_fr3.set_title("Train Error")
		
		plt.tight_layout()


	#plot some samples from the testdata 
	fig3,ax3 = plt.subplots(4,16,figsize=(16,4),sharey="row")
	for i in range(16):
		# plot the input
		ax3[0][i].matshow(test_data[i:i+1].reshape(28,28))
		# plot the probs of visible layer
		ax3[1][i].matshow(DBM.probs[i:i+1].reshape(28,28))
		# plot the recunstructed image
		ax3[2][i].matshow(DBM.rec[i:i+1].reshape(28,28))
		# plot the hidden layer h2 and h1
		# ax3[3][i].matshow(DBM.h1_test[i:i+1].reshape(int(sqrt(DBM.shape[1])),int(sqrt(DBM.shape[1]))))
		ax3[3][i].bar(range(10),DBM.h2_test[i])
		#plot the reconstructed layer h1
		# ax3[5][i].matshow(DBM.rec_h1[i:i+1].reshape(int(sqrt(DBM.shape[1])),int(sqrt(DBM.shape[1]))))
		# plt.matshow(random_recon.reshape(28,28))
	plt.tight_layout(pad=0.0)

	#plot only one digit
	fig4,ax4 = plt.subplots(4,10,figsize=(16,4),sharey="row")
	m=0
	for i in index_for_number[0:10]:
		# plot the input
		ax4[0][m].matshow(test_data[i:i+1].reshape(28,28))
		# plot the probs of visible layer
		ax4[1][m].matshow(DBM.probs[i:i+1].reshape(28,28))
		# plot the recunstructed image
		ax4[2][m].matshow(DBM.rec[i:i+1].reshape(28,28))
		# plot the hidden layer h2 and h1
		# ax4[3][m].matshow(DBM.h1_test[i:i+1].reshape(int(sqrt(DBM.shape[1])),int(sqrt(DBM.shape[1]))))
		ax4[3][m].bar(range(10),DBM.h2_test[i])
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
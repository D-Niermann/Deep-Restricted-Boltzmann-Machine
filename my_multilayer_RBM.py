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
	# workdir="/home/dario/Downloads/DBM Project"
	mpl.rcParams["image.cmap"] = "jet"
	mpl.rcParams["grid.linewidth"] = 0.0

	os.chdir(workdir)
	from Logger import *
	from RBM_Functions import *
	### import seaborn? ###
	if 0:
		import seaborn

		seaborn.set(font_scale=1.4)
		seaborn.set_style("ticks",
			{
			'axes.grid':            False,
			'grid.linestyle':       u':',
			'legend.numpoints':     1,
			'legend.scatterpoints': 1,
			'axes.linewidth':       1,
			'xtick.direction':      'in',
			'ytick.direction':      'in',
			'xtick.major.size': 	5,
			'xtick.minor.size': 	1.0,
			'legend.frameon':       True,
			'ytick.major.size': 	5,
			'ytick.minor.size': 	1.0
			})
	# plt.rcParams['image.cmap'] = 'coolwarm'
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
additional_args=sys.argv
#clear the first entry becaust that is always just the file name
if len(additional_args)>0:
	additional_args.pop(0)

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
		
		self.init_state   = 0
		self.exported     = 0
		self.m            = 0 #laufvariable
		self.num_of_skipped = 100 # how many tf.array value adds get skipped 

		self.log_list=[	["n_units_first_layer",shape[0]],
					["n_units_second_layer",shape[1]],
					["n_units_third_layer",shape[2]],
					["epochs_pretrain",pretrain_epochs],
					["epochs_train",dbm_epochs],
					["batches_pretrain",num_batches_pretrain],
					["batches_dbm_train",dbm_batches],
					["learnrate_pretrain",rbm_learnrate],
					["learnrate_dbm_train",dbm_learnrate],
					["learnrate_dbm_train_end",dbm_learnrate_end],
					["Temperature",temp],
					["pathsuffix_pretrained",pathsuffix_pretrained],
					["pathsuffix",pathsuffix]
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
			sess.run(tf.global_variables_initializer())
			
			#iterate through the RBMs , each iteration is a RBM
			if pre_training:	
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

	################################################################################################################################################
	####  DBM Graph 
	################################################################################################################################################
	def graph_init(self,train_graph):
		""" sets the graph up and loads the pretrained weights in , these are given
		at class definition
		train_grpah :: 1 if the graph is used in training - this will set h2 to placeholder for the label data
				   0 if the graph is used in testing - this will set h2 to a random value outside this function and to be calculated from h1 
		"""
		log.out("Initializing graph")
		self.v  = tf.placeholder(tf.float32,[None,self.shape[0]],name="Visible-Layer") # has self.shape [number of images per batch,number of visible units]
		

		
		if train_graph:
			self.m_tf = tf.placeholder(tf.int32,[],name="running_array_index")
			self.h2   = tf.placeholder(tf.float32,[None,self.shape[2]])

			self.h1_activity_ = tf.Variable(tf.zeros([self.num_of_updates/self.num_of_skipped]))
			self.h2_activity_ = tf.Variable(tf.zeros([self.num_of_updates/self.num_of_skipped]))
			self.train_error_ = tf.Variable(tf.zeros([self.num_of_updates/self.num_of_skipped]))
			self.w1_mean_     = tf.Variable(tf.zeros([self.num_of_updates/self.num_of_skipped]))
			#self.CD1_mean_    = tf.Variable(tf.zeros([self.num_of_updates/self.num_of_skipped]))
			#self.CD2_mean_    = tf.Variable(tf.zeros([self.num_of_updates/self.num_of_skipped]))		


		# self.h2      = tf.placeholder(tf.random_uniform([None,self.shape[1].hidden_units],minval=-1e-3,maxval=1e-3),name="h2_placeholder")

		self.w1 = tf.Variable(self.weights[0],name="Weights1")# init with small random values to break symmetriy
		self.w2 = tf.Variable(self.weights[1],name="Weights2")# init with small random values to break symmetriy

		self.bias_v  = tf.Variable(tf.zeros([self.shape[0]]),name="Visible-Bias")
		self.bias_h1 = tf.Variable(tf.zeros([self.shape[1]]), name="Hidden-Bias")
		self.bias_h2 = tf.Variable(tf.zeros([self.shape[2]]), name="Hidden-Bias")


		### Propagation
		## Forward Feed
		# h1 gets both inputs from h2 and v
		self.h1_prob = sigmoid(tf.matmul(self.v,self.w1) + tf.matmul(self.h2,tf.transpose(self.w2)) + self.bias_h1,temp)
		self.h1      = tf.nn.relu(tf.sign(self.h1_prob - tf.random_uniform(tf.shape(self.h1_prob)))) 
		# h2 only from h1
		if not train_graph:
			self.h2_prob = sigmoid(tf.matmul(self.h1,self.w2) + self.bias_h2,temp)
			self.h2      = tf.nn.relu(tf.sign(self.h2_prob - tf.random_uniform(tf.shape(self.h2_prob)))) 

		## Backward Feed
		self.h1_recon_prob = sigmoid(tf.matmul(self.v,self.w1)+tf.matmul(self.h2,tf.transpose(self.w2))+self.bias_h1, temp)
		self.h1_recon      = tf.nn.relu(tf.sign(self.h1_recon_prob - tf.random_uniform(tf.shape(self.h1_recon_prob)))) 
		self.v_recon_prob  = sigmoid(tf.matmul(self.h1_recon,tf.transpose(self.w1))+self.bias_v, temp)
		self.v_recon       = tf.nn.relu(tf.sign(self.v_recon_prob - tf.random_uniform(tf.shape(self.v_recon_prob)))) 

		## Gibbs step probs
		self.h1_gibbs = sigmoid(tf.matmul(self.v_recon,self.w1) + tf.matmul(self.h2,tf.transpose(self.w2)) + self.bias_h1,temp)
		self.h2_gibbs = sigmoid(tf.matmul(self.h1_recon,self.w2), temp)
		
		#Error and stuff
		self.error  = tf.reduce_mean(tf.square(self.v-self.v_recon))
		self.h1_sum = tf.reduce_sum(self.h1)
		self.h2_sum = tf.reduce_sum(self.h2)

		### Training with contrastive Divergence
		#first weight matrix
		self.pos_grad1  = tf.matmul(tf.transpose(self.v),self.h1)
		self.neg_grad1  = tf.matmul(tf.transpose(self.v_recon),self.h1_gibbs)
		self.numpoints1 = tf.cast(tf.shape(self.v)[0],tf.float32) #number of train inputs per batch (for averaging the CD matrix -> see practical paper by hinton)
		self.CD1        = (self.pos_grad1 - self.neg_grad1)/self.numpoints1
		self.CD1_mean   = tf.reduce_mean(tf.square(self.CD1))
		self.update_w1  = self.w1.assign(self.w1+self.learnrate*self.CD1)
		self.mean_w1    = tf.reduce_mean(tf.square(self.w1))
		# second weight matrix
		self.pos_grad2  = tf.matmul(tf.transpose(self.h1), self.h2)
		self.neg_grad2  = tf.matmul(tf.transpose(self.h1_recon), self.h2_gibbs)
		self.numpoints2 = tf.cast(tf.shape(self.h2)[0],tf.float32) #number of train inputs per batch (for averaging the CD matrix -> see practical paper by hinton)
		self.CD2	    = (self.pos_grad2 - self.neg_grad2)/self.numpoints2
		self.CD2_mean   = tf.reduce_mean(tf.square(self.CD2))
		self.update_w2  = self.w2.assign(self.w2+self.learnrate*self.CD2)
		# bias updates
		self.update_bias_h1 = self.bias_h1.assign(self.bias_h1+self.learnrate*tf.reduce_mean(self.h1-self.h1_gibbs,0))
		self.update_bias_h2 = self.bias_h2.assign(self.bias_h2+self.learnrate*tf.reduce_mean(self.h2-self.h2_gibbs,0))
		self.update_bias_v  = self.bias_v.assign(self.bias_v+self.learnrate*tf.reduce_mean(self.v-self.v_recon,0))
		
		
		if train_graph:
			self.assign_arrays =	[tf.scatter_update(self.train_error_,self.m_tf,self.error),
							 #tf.scatter_update(self.CD1_mean_,self.m_tf,self.CD1_mean),
							 #tf.scatter_update(self.CD2_mean_,self.m_tf,self.CD2_mean),
							 tf.scatter_update(self.w1_mean_,self.m_tf,self.mean_w1),
							 tf.scatter_update(self.h1_activity_,self.m_tf,self.h1_sum),
							 tf.scatter_update(self.h2_activity_,self.m_tf,self.h2_sum),
							]

		### reverse feed
		self.h2_rev      = tf.placeholder(tf.float32,[None,10],name="reverse_h2")
		self.h1_rev_prob = sigmoid(tf.matmul(self.h2_rev, tf.transpose(self.w2))+self.bias_h1,temp)
		self.h1_rev      = tf.nn.relu(tf.sign(self.h1_rev_prob - tf.random_uniform(tf.shape(self.h1_rev_prob)))) 
		self.v_rev_prob  = sigmoid(tf.matmul(self.h1_rev, tf.transpose(self.w1))+self.bias_v,temp)
		self.v_rev       = tf.nn.relu(tf.sign(self.v_rev_prob - tf.random_uniform(tf.shape(self.v_rev_prob)))) 

		sess.run(tf.global_variables_initializer())
		self.init_state=1


	def train(self,epochs,num_batches,cont):
		""" training the DBM with given h2 as labels """
		# init all vars for training
		batchsize      = int(55000/num_batches)
		num_of_updates = epochs*num_batches
		self.num_of_updates = num_of_updates
		d_learnrate    = float(dbm_learnrate_end-self.learnrate)/num_of_updates
		self.m=0
		
		self.h2        = tf.Variable(tf.random_uniform([batchsize,self.shape[2]],minval=-1e-3,maxval=1e-3),name="h2")
		tf.variables_initializer([self.h2], name='init_train')
			

		
		if load_from_file:
			# load data from the file
			self.load_from_file(workdir+"/data/"+pathsuffix)
			self.graph_init(1)
			self.import_()

		# if no files loaded then init the graph with pretrained vars
		if self.init_state==0:
			self.graph_init(1)

		if cont:
			self.graph_init(1)
			self.import_()


		if self.liveplot:
			log.info("Liveplot is on!")
			fig,ax=plt.subplots(1,1,figsize=(15,10))

		
		# starting the training
		for epoch in range(epochs):
			log.start("Deep BM Epoch:",epoch+1,"/",epochs)

			for start, end in zip( range(0, len(train_data), batchsize), range(batchsize, len(train_data), batchsize)):
				
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
					sess.run([self.assign_arrays],
							feed_dict={	self.v:batch,
									self.h2:batch_label,
									self.m_tf:self.m/self.num_of_skipped}
						)
				
				# increase the learnrate
				self.learnrate+=d_learnrate

				self.m+=1

				
				if self.liveplot and plt.fignum_exists(fig.number) and start%40==0:
					ax.cla()
					matrix_new=tile_raster_images(X=self.w1.eval().T, img_shape=(28, 28), tile_shape=(12, 12), tile_spacing=(0,0))
					ax.matshow(matrix_new)
					plt.pause(0.00001)



			# self.train_error_np=self.train_error_.eval()
			# log.out("error:",np.round(self.train_error_np[m],4)," learnrate:",self.learnrate)
			
			log.end()
		log.reset()

		# normalize the activity arrays
		self.h1_activity_*=1./(n_second_layer*batchsize)

		self.export()


	def test(self,test_data):
		""" testing runs without giving h2 , only v is given and h2 has to be infered 
		by the DBM """
		#init the vars and reset the weights and biases 
		log.start("Testing DBM")
		
		self.h2   = tf.Variable(tf.random_uniform([len(test_data),self.shape[2]],minval=-1e-3,maxval=1e-3),name="h2")
		tf.variables_initializer([self.h2], name='init_train')

		if load_from_file and not training:
			self.load_from_file(workdir+pathsuffix)

		self.graph_init(0) # 0 because this graph creates the testing variables where only v is given, not h
		self.import_()

		self.test_error  = self.error.eval({self.v:test_data})
		self.h1_act_test = self.h1_sum.eval({self.v:test_data})
		self.h2_act_test = self.h2_sum.eval({self.v:test_data})

		self.probs      = self.v_recon_prob.eval({self.v:test_data})
		self.rec        = self.v_recon.eval({self.v:test_data})

		self.rec_h1     = self.h1_recon.eval({self.v:test_data})
		self.h1_test    = self.h1.eval({self.v:test_data})
		self.h2_test    = self.h2_prob.eval({self.v:test_data})


		log.end()
		self.h1_act_test*=1./(n_second_layer*len(test_data))
		self.h2_act_test*=1./(n_third_layer*len(test_data))

		# error of classifivation labels
		self.class_error=np.mean(np.abs(DBM.h2_test-test_label))
		
		# #set the maximum = 1 and the rest 0 		
		# log.out("Taking only the maximum")
		# for i in range(10000):
		# 	DBM.h2_test[i]=DBM.h2_test[i]==DBM.h2_test[i].max()

		log.reset()
		log.info("Reconstr. error: ",np.round(DBM.test_error,5), "learnrate: ",np.round(dbm_learnrate,5))
		log.info("Class error: ",np.round(self.class_error,5))
		log.info("Activations of Neurons: ", np.round(self.h1_act_test,2) , np.round(self.h2_act_test,2))


	def reverse_feed(self,my_input_data):
		""" use this in the test sesion to feed a h2 labeled vector to
		genereate numbers """
		v_rev=self.v_rev_prob.eval({self.h2_rev:my_input_data})
		return v_rev


	def gibbs_sampling(self,v_input,label_input,gibbs_steps,liveplot=1):
		if load_from_file and not training:
			self.load_from_file(workdir+pathsuffix)
		self.h2   = tf.Variable(label_input.astype(np.float32),name="h2")
		tf.variables_initializer([self.h2], name='init_train')

		self.graph_init(train_graph=0)
		self.import_()
		if liveplot:
			log.out("Liveplotting gibbs sampling")
			fig,ax=plt.subplots(1,2)
		#start with label h2 vector as input
		self.v_rev_gibbs=self.v_recon_prob.eval({self.v:v_input})
		#now set self.v_rev_gibbs = v and generate h2
		h2_gibbs=self.h2_prob.eval({self.v:v_input})
		#and generate self.v_rev_gibbs again
		for i in range(gibbs_steps):
			if plt.fignum_exists(fig.number):
				ax[0].cla()
				ax[0].matshow(self.v_rev_gibbs.reshape(28,28))
				ax[1].cla()
				ax[1].matshow(h2_gibbs[0][:-1].reshape(3,3))
				plt.pause(0.0001)
				self.v_rev_gibbs=self.v_recon_prob.eval({self.v:self.v_rev_gibbs})
				h2_gibbs=self.h2_prob.eval({self.v:self.v_rev_gibbs})
			else:
				break

		
		plt.close(fig)


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
		new_path = os.getcwd()+"/data/"+str(time_now)
		os.makedirs(new_path)
		os.chdir(new_path)
		np.savetxt("w1.txt", self.w1_np)
		np.savetxt("w2.txt", self.w2_np)
		np.savetxt("bias1.txt", self.bias1_np)
		np.savetxt("bias2.txt", self.bias2_np)
		np.savetxt("bias3.txt", self.bias3_np)
		log.info("Saved weights and biases to:",new_path)

		if save_all_params:
			with open("logfile.txt","w") as log_file:
				for i in range(len(self.log_list)):
					log_file.write(self.log_list[i][0]+","+str(self.log_list[i][1])+"\n")

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

			log.info("Saved Parameters to:",new_path)


		
		os.chdir(workdir)


####################################################################################################################################
#### User Settings ###

num_batches_pretrain = 500
dbm_batches          = 500
pretrain_epochs      = 1
dbm_epochs           = 5

rbm_learnrate     = 0.005
dbm_learnrate     = 0.01
dbm_learnrate_end = 0.01

temp = 1.

pre_training    = 0 #if no pretrain then files are automatically loaded

training       = 0
plotting       = 0
gibbs_sampling = 1

save_to_file          = 0
save_all_params       = 0
save_pretrained       = 0

load_from_file        = 1
pathsuffix            = "/data/Fri Jan 12 16-40-57 2018"
pathsuffix_pretrained = "Fri Jan 12 11-00-46 2018"


number_of_layers = 3
n_first_layer    = 784
n_second_layer   = 15*15
n_third_layer    = 10



######### DBM ##########################################################################
#### Pre training is ended - create the DBM with the gained weights
# if i == 0,1,2,...: (das ist das i von der echo cluster schleife) in der dbm class stehen dann die parameter für das jeweilige i 
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

			log.end()

	# test session
	with tf.Session() as sess:
		log.start("Test Session")
		# new session for test images - v has 10.000 length 
		#testing the network , this also inits the graph so do not comment it out
		DBM.test(test_data) 

		# make a reverse feed with all 10.000 label - later only a few get plottet 
		v_rev=DBM.reverse_feed(test_label)

		log.end()

if gibbs_sampling:
	with tf.Session() as sess:
		log.start("Gibbs Sampling Session")
		# start a new session because gibbs sampling only has 1 test input - v has 1 length
		DBM.gibbs_sampling(test_data[7:8],test_label[7:8], 200, liveplot=1)
		log.end()

if save_to_file:
	DBM.write_to_file()

####################################################################################################################################
#### Plot
# Plot the Weights, Errors and other informations
if plotting:
	
	map1=plt.matshow(tile_raster_images(X=DBM.w1_np.T, img_shape=(28, 28), tile_shape=(12, 12), tile_spacing=(0,0)))
	plt.title("W 1")


	# map2=plt.matshow(tile_raster_images(X=DBM.w2_np.T, img_shape=(int(sqrt(n_second_layer)),int(sqrt(n_second_layer))), tile_shape=(12, 12), tile_spacing=(0,0)))
	# plt.title("W 2")
	# plt.colorbar(map2)

	if training:
		fig_fr=plt.figure(figsize=(7,9))
		
		ax_fr1=fig_fr.add_subplot(311)
		ax_fr1.plot(DBM.h1_activity_np)
		
		ax_fr2=fig_fr.add_subplot(312)
		# ax_fr2.plot(DBM.CD1_mean_np,label="CD1")
		# ax_fr2.plot(DBM.CD2_mean_np,label="CD2")
		ax_fr2.plot(DBM.w1_mean_np,label="Weights")
		ax_fr1.set_title("Firerate h1 layer")
		ax_fr2.set_title("Weights, CD1 and CD2 mean")
		ax_fr2.legend(loc="best")
		
		ax_fr3=fig_fr.add_subplot(313)
		ax_fr3.plot(DBM.train_error_np,"k")
		ax_fr3.set_title("Train Error")
		
		plt.tight_layout()


	#plot some samples from the testdata 
	fig3,ax3 = plt.subplots(6,15,figsize=(16,4))
	for i in range(15):
		# plot the input
		ax3[0][i].matshow(test_data[i:i+1].reshape(28,28))
		# plot the probs of visible layer
		ax3[1][i].matshow(DBM.probs[i:i+1].reshape(28,28))
		# plot the recunstructed image
		ax3[2][i].matshow(DBM.rec[i:i+1].reshape(28,28))
		# plot the hidden layer h2 and h1
		ax3[3][i].matshow(DBM.h1_test[i:i+1].reshape(int(sqrt(DBM.shape[1])),int(sqrt(DBM.shape[1]))))
		ax3[4][i].matshow(DBM.h2_test[i:i+1,:9].reshape(int(sqrt(DBM.shape[2])),int(sqrt(DBM.shape[2]))))
		#plot the reconstructed layer h1
		ax3[5][i].matshow(DBM.rec_h1[i:i+1].reshape(int(sqrt(DBM.shape[1])),int(sqrt(DBM.shape[1]))))
		# plt.matshow(random_recon.reshape(28,28))
	plt.tight_layout(pad=0.0)

	#plot only one digit
	fig4,ax4 = plt.subplots(6,10,figsize=(16,4))
	m=0
	for i in index_for_number[0:10]:
		# plot the input
		ax4[0][m].matshow(test_data[i:i+1].reshape(28,28))
		# plot the probs of visible layer
		ax4[1][m].matshow(DBM.probs[i:i+1].reshape(28,28))
		# plot the recunstructed image
		ax4[2][m].matshow(DBM.rec[i:i+1].reshape(28,28))
		# plot the hidden layer h2 and h1
		ax4[3][m].matshow(DBM.h1_test[i:i+1].reshape(int(sqrt(DBM.shape[1])),int(sqrt(DBM.shape[1]))))
		ax4[4][m].matshow(DBM.h2_test[i:i+1,:9].reshape(int(sqrt(DBM.shape[2])),int(sqrt(DBM.shape[2]))))
		#plot the reconstructed layer h1
		ax4[5][m].matshow(DBM.rec_h1[i:i+1].reshape(int(sqrt(DBM.shape[1])),int(sqrt(DBM.shape[1]))))
		# plt.matshow(random_recon.reshape(28,28))
		m+=1
	plt.tight_layout(pad=0.0)

	# plot the reverse_feed:
	fig5,ax5 = plt.subplots(2,14,figsize=(16,4))
	for i in range(14):
		ax5[0][i].matshow(test_label[i,:9].reshape(3,3))
		ax5[1][i].matshow(v_rev[i].reshape(28,28))
	plt.tight_layout(pad=0.0)


	plt.matshow(np.mean(v_rev[index_for_number[:]],0).reshape(28,28))

plt.show()

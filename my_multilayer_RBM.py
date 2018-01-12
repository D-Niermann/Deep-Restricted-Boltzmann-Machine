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
	import os,time
	from math import exp,sqrt,sin,pi,cos,log
	np.set_printoptions(precision=3)
	
	workdir="/Users/Niermann/Google Drive/Masterarbeit/Python"
	# workdir="/home/dario/Downloads"
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
		if (test_label[i]==[0,0,0,0,0,0,1,0,0,0]).sum()==10:
			index_for_number.append(i)

	half_images = test_data[0:11]
	#halfing some images from test_data
	half_images[1:6,500:] = 0

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

	def train(self,batch):
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
	shape should be the array with the DBM classes so one can
	access all unit length. as weights the pretrained weights 
	should be used.
	"""

	def __init__(self,shape,weights,learnrate,liveplot):
		self.n_layers     = len(shape)
		self.liveplot     = liveplot
		self.shape        = shape
		self.train_error_ = []
		self.weights      = weights
		self.init_state   = 0
		self.learnrate    = learnrate
		self.exported     = 0

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
		self.v  = tf.placeholder(tf.float32,[None,self.shape[0].visible_units],name="Visible-Layer") # has self.shape [number of images per batch,number of visible units]

		if train_graph:
			self.h2 = tf.placeholder(tf.float32,[None,self.shape[1].hidden_units])

		# self.h2      = tf.placeholder(tf.random_uniform([None,self.shape[1].hidden_units],minval=-1e-3,maxval=1e-3),name="h2_placeholder")

		self.w1 = tf.Variable(self.weights[0],name="Weights1")# init with small random values to break symmetriy
		self.w2 = tf.Variable(self.weights[1],name="Weights2")# init with small random values to break symmetriy

		self.bias_v  = tf.Variable(tf.zeros([self.shape[0].visible_units]),name="Visible-Bias")
		self.bias_h1 = tf.Variable(tf.zeros([self.shape[0].hidden_units]), name="Hidden-Bias")
		self.bias_h2 = tf.Variable(tf.zeros([self.shape[1].hidden_units]), name="Hidden-Bias")


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
		self.v_recon_prob  = sigmoid(tf.matmul(self.h1,tf.transpose(self.w1))+self.bias_v, temp)
		self.v_recon       = tf.nn.relu(tf.sign(self.v_recon_prob - tf.random_uniform(tf.shape(self.v_recon_prob)))) 
		self.h1_recon_prob = sigmoid(tf.matmul(self.h2,tf.transpose(self.w2)), temp)
		self.h1_recon      = tf.nn.relu(tf.sign(self.h1_recon_prob - tf.random_uniform(tf.shape(self.h1_recon_prob)))) 


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
		self.update_w1  = self.w1.assign(self.w1+self.learnrate*self.CD1)
		# second weight matrix
		self.pos_grad2  = tf.matmul(tf.transpose(self.h1), self.h2)
		self.neg_grad2  = tf.matmul(tf.transpose(self.h1_recon), self.h2_gibbs)
		self.numpoints2 = tf.cast(tf.shape(self.h2)[0],tf.float32) #number of train inputs per batch (for averaging the CD matrix -> see practical paper by hinton)
		self.CD2	    = (self.pos_grad2 - self.neg_grad2)/self.numpoints2
		self.update_w2  = self.w2.assign(self.w2+self.learnrate*self.CD2)
		# bias updates
		self.update_bias_h1 = self.bias_h1.assign(self.bias_h1+self.learnrate*tf.reduce_mean(self.h1-self.h1_gibbs,0))
		self.update_bias_h2 = self.bias_h2.assign(self.bias_h2+self.learnrate*tf.reduce_mean(self.h2-self.h2_gibbs,0))
		self.update_bias_v  = self.bias_v.assign(self.bias_v+self.learnrate*tf.reduce_mean(self.v-self.v_recon,0))
		sess.run(tf.global_variables_initializer())

		### reverse feed
		self.h2_rev = tf.placeholder(tf.float32,[None,10],name="reverse_h2")
		self.h1_rev_prob = sigmoid(tf.matmul(self.h2_rev, tf.transpose(self.w2))+self.bias_h1,temp)
		self.h1_rev      = tf.nn.relu(tf.sign(self.h1_rev_prob - tf.random_uniform(tf.shape(self.h1_rev_prob)))) 
		self.v_rev_prob  = sigmoid(tf.matmul(self.h1_rev, tf.transpose(self.w1))+self.bias_v,temp)
		self.v_rev       = tf.nn.relu(tf.sign(self.v_rev_prob - tf.random_uniform(tf.shape(self.v_rev_prob)))) 

		self.init_state=1


	def train(self,epochs,num_batches):
		""" training the DBM with given h2 as labels """
		# init all vars for training
		batchsize      = int(55000/num_batches)
		num_of_updates = epochs*num_batches
		self.h2        = tf.Variable(tf.random_uniform([batchsize,self.shape[1].hidden_units],minval=-1e-3,maxval=1e-3),name="h2")
		tf.variables_initializer([self.h2], name='init_train')

		if load_from_file:
			# load data from the file
			self.load_from_file(workdir+pathsuffix)
			self.graph_init(1)
			self.import_()

		# if no files loaded then init the graph with pretrained vars
		if self.init_state==0:
			self.graph_init(1)

		if self.liveplot:
			log.info("Liveplot is on!")
			fig,ax=plt.subplots(1,1,figsize=(15,10))

		m=0 #laufvariable
		
		self.h1_activity=np.zeros(num_of_updates)
		self.h2_activity=np.zeros(num_of_updates)
		self.train_error_=np.zeros(num_of_updates)
		# starting the training
		for epoch in range(epochs):
			log.start("Deep BM Epoch:",epoch+1,"/",epochs)

			for start, end in zip( range(0, len(train_data), batchsize), range(batchsize, len(train_data), batchsize)):
				m+=1
				# define a batch
				batch = train_data[start:end]
				batch_label = train_label[start:end]
				# run all updates 
				sess.run([self.update_w1,self.update_w2,self.update_bias_v,self.update_bias_h1,self.update_bias_h2],feed_dict={self.v:batch,self.h2:batch_label})
				# append error and other data to self variables
				self.train_error_[m]=(self.error.eval({self.v:batch,self.h2:batch_label}))

				self.h1_activity[m]=sess.run(self.h1_sum,feed_dict={self.v:batch,self.h2:batch_label})
				self.h2_activity[m]=sess.run(self.h2_sum,feed_dict={self.v:batch,self.h2:batch_label})
				
				if self.liveplot and plt.fignum_exists(fig.number) and start%40==0:
						ax.cla()
						matrix_new=tile_raster_images(X=self.w1.eval().T, img_shape=(28, 28), tile_shape=(12, 12), tile_spacing=(0,0))
						ax.matshow(matrix_new)
						plt.pause(0.00001)

			
			log.out("Error:",np.round(self.train_error_[m],4))
			log.end()
		log.reset()

		# normalize the activity arrays
		self.h1_activity*=1./(n_second_layer*batchsize)
		self.h2_activity*=1./(n_third_layer*batchsize)

		self.export()



	def test(self,test_data):
		""" testing runs without giving h2 , only v is given and h2 has to be infered 
		by the DBM """
		#init the vars and reset the weights and biases 
		log.start("Testing DBM")
		
		self.h2   = tf.Variable(tf.random_uniform([len(test_data),self.shape[1].hidden_units],minval=-1e-3,maxval=1e-3),name="h2")
		tf.variables_initializer([self.h2], name='init_train')

		if load_from_file and not training:
			self.load_from_file(workdir+pathsuffix)

		elif training:
			self.graph_init(0) # 0 because this graph creates the testing variables where only v is given, not h
			self.import_()
		
			### starting the testing
			self.test_error  = self.error.eval({self.v:test_data})
			self.h1_act_test = self.h1_sum.eval({self.v:test_data})
			self.h2_act_test = self.h2_sum.eval({self.v:test_data})

			self.probs      = self.v_recon_prob.eval({self.v:test_data})
			self.rec        = self.v_recon.eval({self.v:test_data})

			self.rec_h1     = self.h1_recon.eval({self.v:test_data})
			self.h1_test    = self.h1.eval({self.v:test_data})
			self.h2_test    = self.h2_prob.eval({self.v:test_data})

		else:
			log.info("Neither loaded from file nor trained DBM cant perform anyting")

		log.end()
		self.h1_act_test*=1./(n_second_layer*len(test_data))
		self.h2_act_test*=1./(n_third_layer*len(test_data))

		log.reset()
		log.info("test error: ",(DBM.test_error), "learnrate: ",dbm_learnrate)
		log.info("Activations of Neurons: ", np.round(self.h1_act_test,2) , np.round(self.h2_act_test,2))

	def reverse_feed(self,input_data):
		""" use this in the test sesion to feed a h2 labeled vector to
		genereate numbers """
		v_rev=self.v_rev.eval({self.h2_rev:input_data})
		return v_rev


	def gibbs_sampling(self,input,gibbs_steps,liveplot=1):
		pass

	def export(self):
		# carefull: w1 is tf vriable and numpy array !
		self.w1_np    = self.w1.eval() 
		self.w2_np    = self.w2.eval()
		self.bias1_np = self.bias_v.eval()
		self.bias2_np = self.bias_h1.eval()
		self.bias3_np = self.bias_h2.eval()
		self.exported = 1

	def write_to_file(self):
		if self.exported!=1:
			self.export()
		new_path = os.getcwd()+"/"+str(time_now)
		os.makedirs(new_path)
		os.chdir(new_path)
		np.savetxt("w1.txt", self.w1_np)
		np.savetxt("w2.txt", self.w2_np)
		np.savetxt("bias1.txt", self.bias1_np)
		np.savetxt("bias2.txt", self.bias2_np)
		np.savetxt("bias3.txt", self.bias3_np)
		log.info("Saved weights and biases to:",new_path)
		os.chdir(workdir)


####################################################################################################################################
#### User Settings ###
num_batches_pretrain = 1000
dbm_batches          = 1000
pretrain_epochs      = 1
dbm_epochs           = 3

rbm_learnrate = 0.001
dbm_learnrate = 0.01

temp          = 1.

pre_training = 0 #if no pretrain then files are automatically loaded
training     = 1
plotting     = 1

save_to_file    = 0
save_pretrained = 0
load_from_file  = 1
pathsuffix      = "/Fri Jan 12 11-25-58 2018"
pathsuffix_pretrained = "Fri Jan 12 11-00-46 2018"


number_of_layers = 3
n_first_layer    = 784
n_second_layer   = 15*15
n_third_layer    = 10


####################################################################################################################################
#### Create RBMs and merge them into one list for iteration###
#RBM(visible units, hidden units, forw_mult, back_mult, liveplot,...)
RBMs    = [0]*(number_of_layers-1)
RBMs[0] = RBM(n_first_layer, n_second_layer , 2, 1, learnrate=rbm_learnrate, liveplot=0)
RBMs[1] = RBM(n_second_layer, n_third_layer , 1, 2, learnrate=rbm_learnrate, liveplot=0)




####################################################################################################################################
#### Session ####
log.reset()
log.info(time_now)

# d_learnrate   = float(learnrate_max-learnrate)/num_of_updates


#define a figure for liveplotting
if pre_training:
	for rbm in RBMs:
		if rbm.liveplot:
			log.info("Liveplot is open!")
			fig,ax=plt.subplots(1,1,figsize=(15,10))
			break

batchsize_pretrain = int(55000/num_batches_pretrain)

# start the session for training
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	#iterate through the RBMs , each iteration is a RBM
	if pre_training:	
		for RBM_i,RBM in enumerate(RBMs):
			log.start("Pretraining ",str(RBM_i+1)+".", "RBM")

			for epoch in range(pretrain_epochs):
				log.start("Epoch:",epoch+1,"/",pretrain_epochs)
				
				for start, end in zip( range(0, len(train_data), batchsize_pretrain), range(batchsize_pretrain, len(train_data), batchsize_pretrain)):
					#### define a batch
					batch = train_data[start:end]
					# train the rbm 
					w_i,error_i = RBM.train(batch)
					#### liveplot
					if RBM.liveplot and plt.fignum_exists(fig.number) and start%40==0:
						ax.cla()
						rbm_shape=int(sqrt(RBM.visible_units))
						matrix_new=tile_raster_images(X=w_i.T, img_shape=(rbm_shape, rbm_shape), tile_shape=(10, 10), tile_spacing=(0,0))
						ax.matshow(matrix_new)
						plt.pause(0.00001)


				log.info("Learnrate:",round(rbm_learnrate,4))
				log.info("Error",round(error_i,4))
				log.end() #ending the epoch


			log.end() #ending training the rbm 
			log.reset()
			#append self.error array here 

		weights=[]
		for i in range(len(RBMs)):
			weights.append(RBMs[i].w.eval())

		if save_pretrained:
			for i in range(len(weights)):
				np.savetxt("Pretrained-"+" %i "%i+str(time_now)+".txt", weights[i])
	else:
		weights=[]
		log.out("Loading Pretrained from file")
		for i in range(number_of_layers-1):
			weights.append(np.loadtxt("Pretrained-"+" %i "%i+pathsuffix_pretrained+".txt").astype(np.float32))


	######### DBM ##########################################################################
	#### Pre training is ended - create the DBM with the gained weights
	DBM = DBM_class(RBMs, weights,learnrate=dbm_learnrate,liveplot=0)

	if training:
		DBM.train(epochs=dbm_epochs, num_batches=dbm_batches)

# test session
with tf.Session() as sess:

	#testing the network
	DBM.test(test_data) 

	v_rev=DBM.reverse_feed(test_label)

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


	fig_fr=plt.figure("Fire rates")
	ax_fr1=fig_fr.add_subplot(211)
	ax_fr1.plot(DBM.h1_activity[::4])
	ax_fr2=fig_fr.add_subplot(212)
	ax_fr2.plot(DBM.h2_activity[::4])
	ax_fr1.set_title("Firerate h1 layer")
	ax_fr2.set_title("Firerate h2 layer")
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
		ax3[3][i].matshow(DBM.h1_test[i:i+1].reshape(int(sqrt(DBM.shape[0].hidden_units)),int(sqrt(DBM.shape[0].hidden_units))))
		ax3[4][i].matshow(DBM.h2_test[i:i+1,:9].reshape(int(sqrt(DBM.shape[1].hidden_units)),int(sqrt(DBM.shape[1].hidden_units))))
		#plot the reconstructed layer h1
		ax3[5][i].matshow(DBM.rec_h1[i:i+1].reshape(int(sqrt(DBM.shape[0].hidden_units)),int(sqrt(DBM.shape[0].hidden_units))))
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
		ax4[3][m].matshow(DBM.h1_test[i:i+1].reshape(int(sqrt(DBM.shape[0].hidden_units)),int(sqrt(DBM.shape[0].hidden_units))))
		ax4[4][m].matshow(DBM.h2_test[i:i+1,:9].reshape(int(sqrt(DBM.shape[1].hidden_units)),int(sqrt(DBM.shape[1].hidden_units))))
		#plot the reconstructed layer h1
		ax4[5][m].matshow(DBM.rec_h1[i:i+1].reshape(int(sqrt(DBM.shape[0].hidden_units)),int(sqrt(DBM.shape[0].hidden_units))))
		# plt.matshow(random_recon.reshape(28,28))
		m+=1
	plt.tight_layout(pad=0.0)
	
	# plot the reverse_feed:
	fig5,ax5 = plt.subplots(2,14,figsize=(16,4))
	for i in range(14):
		ax5[0][i].matshow(test_label[i,:9].reshape(3,3))
		ax5[1][i].matshow(v_rev[i].reshape(28,28))
	plt.tight_layout(pad=0.0)

plt.show()
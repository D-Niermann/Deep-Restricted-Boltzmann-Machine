#### Imports
if True:
	# -*- coding: utf-8 -*-
	print "Starting..."
	import numpy as np
	import numpy.random as rnd
	import matplotlib.pyplot as plt
	import matplotlib.image as img
	import tensorflow as tf
	# import scipy.ndimage.filters as filters
	# import pandas as pd
	import os,time,sys
	from math import exp,sqrt,sin,pi,cos,log
	np.set_printoptions(precision=3)
	# plt.style.use('ggplot')
	os.chdir("/Users/Niermann/Google Drive/Masterarbeit/Python")
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


#### Load MNIST Data 
if "train_data" not in globals():
	log.out("Loading Data")
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	train_data, trY, test_data, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

	#get test data of only one number class:
	index_for_number=[]
	for i in range(len(teY)):
		if (teY[i]==[0,0,0,0,0,0,1,0,0,0]).sum()==10:
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

	def test():
		pass


################################################################################################################################################
### Class Deep BM 
class DBM_class(object):
	"""defines a deep boltzmann machine
	shape should be the array with the DBM classes so one can
	access all unit length. as weights the pretrained weights 
	should be used.
	"""

	def __init__(self,shape,weights,learnrate):
		self.n_layers = len(shape)
		self.shape = shape
		self.train_error_ = []
		self.weights=weights
		self.init_state=0
		self.learnrate=learnrate

	def graph_init(self):
		################################################################################################################################################
		####  DBM Graph
		################################################################################################################################################
		""" shape definitions are wired here-normally one would define [rows,columns] but here it is reversed with [columns,rows]? """

		self.v        = tf.placeholder(tf.float32,[None,self.shape[0].visible_units],name="Visible-Layer") # has self.shape [number of images per batch,number of visible units]
		

		# self.h2      = tf.Variable(tf.random_uniform([len(batch_data),self.shape[1].hidden_units],minval=-1e-3,maxval=1e-3),name="h2")

		self.w1       = tf.Variable(self.weights[0],name="Weights1")# init with small random values to break symmetriy
		self.bias_v   = tf.Variable(tf.zeros([self.shape[0].visible_units]),name="Visible-Bias")
		self.bias_h1  = tf.Variable(tf.zeros([self.shape[0].hidden_units]),name="Hidden-Bias")

		self.w2      = tf.Variable(self.weights[1],name="Weights2")# init with small random values to break symmetriy
		self.bias_h2 = tf.Variable(tf.zeros([self.shape[1].hidden_units]),name="Hidden-Bias")


		### Propagation
		## Forward Feed
		# h1 gets both inputs from h2 and v
		self.h1_prob = sigmoid(tf.matmul(self.v,self.w1) + tf.matmul(self.h2,tf.transpose(self.w2)) + self.bias_h1,temp)
		self.h1      = tf.nn.relu(tf.sign(self.h1_prob - tf.random_uniform(tf.shape(self.h1_prob)))) 
		# h2 only from h1
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
		
		#Error
		self.error  = tf.reduce_mean(tf.square(self.v-self.v_recon))

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

		self.init_state=1


	def train(self,epochs,num_batches):
		# init all vars for training
		batchsize      = 55000/num_batches
		self.batchsize = batchsize
		num_of_updates = epochs*num_batches
		self.h2        = tf.Variable(tf.random_uniform([batchsize,self.shape[1].hidden_units],minval=-1e-3,maxval=1e-3),name="h2")
		tf.variables_initializer([self.h2], name='init_train')
		if self.init_state==0:
			self.graph_init()


		for epoch in range(epochs):
			log.start("Deep BM Epoch:",epoch+1,"/",epochs)

			for start, end in zip( range(0, len(train_data), batchsize), range(batchsize, len(train_data), batchsize)):
				# define a batch
				batch = train_data[start:end]
				# run all updates 
				sess.run([self.update_w1,self.update_w2,self.update_bias_v,self.update_bias_h1,self.update_bias_h2],feed_dict={self.v:batch})
				# append error and other data to self variables
				self.train_error_.append(self.error.eval({self.v:batch}))
			
			log.info("Error:",self.train_error_[-1])
			
			log.end()


	def test(self):
		#init the vars
		self.h2      = tf.Variable(tf.random_uniform([len(test_data),self.shape[1].hidden_units],minval=-1e-3,maxval=1e-3),name="h2")
		tf.variables_initializer([self.h2], name='init_train')
		if self.init_state==0:
			self.graph_init()
		
		""" testing only the batch because shapes will not match when taking a bigger batch """
		self.test_batch=test_data[0:self.batchsize]

		self.test_error =self.error.eval({self.v:self.test_batch})

		self.probs      = DBM.v_recon_prob.eval({DBM.v:self.test_batch})
		self.rec        = DBM.v_recon.eval({DBM.v:self.test_batch})


	def export(self):
		self.w1    = tile_raster_images(X=DBM.w1.eval().T, img_shape=(28, 28), tile_shape=(10, 10), tile_spacing=(0,0))
		self.w2    = DBM.w2.eval()
		self.bias1 = DBM.bias_v.eval()
		self.bias2 = DBM.bias_h1.eval()
		self.bias3 = DBM.bias_h2.eval()



####################################################################################################################################
#### User Settings ###
num_batches_pretrain = 500
dbm_batches          = 500
pretrain_epochs      = 1
dbm_epochs           = 4

dbm_learnrate        = 0.005
rbm_learnrate        = 0.005

temp                 = 1.0

pre_training   = 1
training       = 1
plotting       = 1

save_to_file   = 0
load_from_file = 0
file_suffix    = ""

number_of_layers = 3
n_first_layer    = 784
n_second_layer   = 15*15
n_third_layer    = 100
####################################################################################################################################
#### Create RBMs and merge them into one list for iteration###
#RBM(visible units, hidden units, forw_mult, back_mult, liveplot,...)
RBMs    = [0]*(number_of_layers-1)
RBMs[0] = RBM(n_first_layer, n_second_layer , 2, 1, learnrate=rbm_learnrate, liveplot=0)
RBMs[1] = RBM(n_second_layer, n_third_layer , 1, 2, learnrate=rbm_learnrate, liveplot=0)
####################################################################################################################################
#### Session ####
time_now = time.asctime()
log.reset()

log.info(time_now)

errors         = []
mean_w_        = []
update         = 0
batchsize_pretrain = 55000/num_batches_pretrain


# d_learnrate   = float(learnrate_max-learnrate)/num_of_updates


#define a figure for liveplotting
if pre_training:
	for rbm in RBMs:
		if rbm.liveplot:
			log.info("Liveplot is open!")
			fig,ax=plt.subplots(1,1,figsize=(15,10))
			break
	

# start the session
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	# if load_from_file:
	# 	sess.run(w.assign(init_pretrained(file_suffix,w=1)[0]))
	# 	sess.run(bias_v.assign(init_pretrained(file_suffix,bias_v=1)[0]))
	# 	sess.run(bias_h.assign(init_pretrained(file_suffix,bias_h=1)[0]))

	#iterate through the RBMs , each iteration is a RBM
	if pre_training:	
		for RBM_i,RBM in enumerate(RBMs):
			log.start("Pretraining ",str(RBM_i+1)+".", "RBM")

			for epoch in range(pretrain_epochs):
				log.start("Epoch:",epoch+1,"/",pretrain_epochs)
				
				for start, end in zip( range(0, len(train_data), batchsize_pretrain), range(batchsize_pretrain, len(train_data), batchsize_pretrain)):
					#### define a batch
					batch = train_data[start:end]

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

	######### DBM ##########################################################################
	#### Pre training is ended - create the DBM with the gained weights
	DBM = DBM_class(RBMs, [RBMs[0].w.eval(),RBMs[1].w.eval()],learnrate=dbm_learnrate)
	
	if training:
		DBM.train(epochs=dbm_epochs, num_batches=dbm_batches)
		DBM.export()
		
	DBM.test()				
	
log.reset()
log.info("test error: ",(DBM.test_error), "learnrate: ",dbm_learnrate)

# Savin to file
if save_to_file:
	save(str(error_i),w_i,ubv,ubh)


####################################################################################################################################
#### Plot
# Plot the Weights, Errors and other informations
if plotting:
	fig_w,ax_w=plt.subplots(2,1,figsize=(10,10))
	map1=ax_w[0].matshow(DBM.w1)
	ax_w[0].set_title("W 1")
	plt.colorbar(map1,ax_w[1])
	
	map2=ax_w[1].matshow(DBM.w2.T)
	ax_w[1].set_title("W 2")
	plt.colorbar(map2,ax_w[1])


	fig3,ax3 = plt.subplots(3,10,figsize=(16,4))
	for i in range(10):
		# plot the input
		ax3[0][i].matshow(DBM.test_batch[i:i+1].reshape(28,28))
		# plot the probs of visible layer
		ax3[1][i].matshow(DBM.probs[i:i+1].reshape(28,28))
		# plot the recunstructed image
		ax3[2][i].matshow(DBM.rec[i:i+1].reshape(28,28))
		# plt.matshow(random_recon.reshape(28,28))



	plt.show()
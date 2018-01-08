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

	def __init__(self,vu,hu,forw_mult,back_mult,liveplot):
		#### User Variables

		self.hidden_units  = hu
		self.visible_units = vu


		self.liveplot      = liveplot

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
		self.update_w = self.w.assign(self.w+learnrate*self.CD)
		self.mean_w   = tf.reduce_mean(self.w)
		#update bias
		""" Since vectors v and h are actualy matrices with number of batch_size images in them, reduce mean will make them to a vector again """
		self.update_bias_v = self.bias_v.assign(self.bias_v+learnrate*tf.reduce_mean(self.v-self.v_recon,0))
		self.update_bias_h = self.bias_h.assign(self.bias_h+learnrate*tf.reduce_mean(self.h-self.h_gibbs,0))


		# reverse feed
		self.h_rev       = tf.placeholder(tf.float32,[None,self.hidden_units],name="Reverse-hidden")
		self.v_prob_rev  = sigmoid(tf.matmul(self.h_rev,tf.transpose(self.w)) + self.bias_v,temp)
		self.v_recon_rev = tf.nn.relu(tf.sign(self.v_prob_rev - tf.random_uniform(tf.shape(self.v_prob_rev))))

class DBM_class(object):
	"""defines a deep boltzmann machine
	shape should be the array with the DBM classes so one can
	access all unit length. as weights the pretrained weights 
	should be used.
	"""

	def __init__(self,shape,weights):


		self.n_layers = len(shape)
		self.shape = shape

		################################################################################################################################################
		####  DBM Graph
		################################################################################################################################################
		""" shape definitions are wired here-normally one would define [rows,columns] but here it is reversed with [columns,rows]? """

		self.v        = tf.placeholder(tf.float32,[None,shape[0].visible_units],name="Visible-Layer") # has shape [number of images per batch,number of visible units]

		self.w1       = tf.Variable(weights[0],name="Weights1")# init with small random values to break symmetriy
		self.bias_v   = tf.Variable(tf.zeros([shape[0].visible_units]),name="Visible-Bias")
		self.bias_h1  = tf.Variable(tf.zeros([shape[0].hidden_units]),name="Hidden-Bias")

		self.h2      = tf.Variable(tf.random_uniform([batchsize,shape[1].hidden_units],minval=-1e-3,maxval=1e-3),name="h2")
		self.w2      = tf.Variable(weights[1],name="Weights2")# init with small random values to break symmetriy
		self.bias_h2 = tf.Variable(tf.zeros([shape[1].hidden_units]),name="Hidden-Bias")


		### Propagation
		## Forward Feed
		# h1 gets both inputs from h2 and v
		self.h1_prob = sigmoid(tf.matmul(self.v,self.w1) + tf.matmul(self.h2,tf.transpose(self.w2)) + self.bias_h1,temp)
		self.h1      = tf.nn.relu(tf.sign(self.h1_prob - tf.random_uniform(tf.shape(self.h1_prob)))) 
		# h2 only from h1
		self.h2_prob = sigmoid(tf.matmul(self.h1,self.w2) + self.bias_h2,temp)
		self.h2      = tf.nn.relu(tf.sign(self.h2_prob - tf.random_uniform(tf.shape(self.h2_prob)))) 

		## Backward Feed
		self.v_recon_prob  = sigmoid(tf.matmul(self.h1,tf.transpose(self.w1)), temp)
		self.v_recon       = tf.nn.relu(tf.sign(self.v_recon_prob - tf.random_uniform(tf.shape(self.v_recon_prob)))) 
		self.h1_recon_prob = sigmoid(tf.matmul(self.h2,tf.transpose(self.w2)), temp)
		self.h1_recon      = tf.nn.relu(tf.sign(self.h1_recon_prob - tf.random_uniform(tf.shape(self.h1_recon_prob)))) 
		# self.h2_recon_prob = sigmoid(tf.matmul(self.h1,tf.transpose(self.w2)), temp)
		# self.h2_recon      = tf.nn.relu(tf.sign(self.h2_recon_prob - tf.random_uniform(tf.shape(self.h2_recon_prob)))) 

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
		self.update_w1  = self.w1.assign(self.w1+learnrate*self.CD1)
		# second weight matrix
		self.pos_grad2  = tf.matmul(tf.transpose(self.h1), self.h2)
		self.neg_grad2  = tf.matmul(tf.transpose(self.h1_recon), self.h2_gibbs)
		self.numpoints2 = tf.cast(tf.shape(self.h2)[0],tf.float32) #number of train inputs per batch (for averaging the CD matrix -> see practical paper by hinton)
		self.CD2	    = (self.pos_grad2 - self.neg_grad2)/self.numpoints2
		self.update_w2  = self.w2.assign(self.w2+learnrate*self.CD2)
		

####################################################################################################################################
#### User Settings ###
num_batches     = 1000
pretrain_epochs = 1
epochs          = 4
learnrate       = 0.005
learnrate_max   = 0.005
temp            = 1.0
pre_training    = 1
training        = 1

save_to_file   = 0
load_from_file = 0
file_suffix    = ""

#### Create RBMs and merge them into one list for iteration###
#RBM(visible units, hidden units, forw_mult, back_mult, liveplot,...)
RBMs    = [0]*2
RBMs[0] = RBM(784, 10*10, 2, 1, 0)
RBMs[1] = RBM(10*10, 100 , 1, 2, 0)



####################################################################################################################################
#### Session ####
time_now = time.asctime()
log.reset()

log.info(time_now)

errors         = []
mean_w_        = []
update         = 0
batchsize      = 55000/num_batches 
num_of_updates = epochs*num_batches
d_learnrate    = float(learnrate_max-learnrate)/num_of_updates


#define a figure for liveplotting
if pre_training:
	fig,ax=plt.subplots(1,1,figsize=(15,10))

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
				log.start("Epoch:",epoch+1,"/",epochs)
				
				for start, end in zip( range(0, len(train_data), batchsize), range(batchsize, len(train_data), batchsize)):
					#### define a batch
					batch = train_data[start:end]

					RBM.my_input_data = batch
					if RBM_i==1:
						RBM.my_input_data=RBMs[RBM_i-1].h.eval({RBMs[RBM_i-1].v:batch})
					elif RBM_i==2:
						RBM.my_input_data=RBMs[RBM_i-1].h.eval({RBMs[RBM_i-1].v:RBMs[RBM_i-1].my_input_data})



					#### update the weights
					RBM.w_i,error_i=sess.run([RBM.update_w,RBM.error],feed_dict={RBM.v:RBM.my_input_data})

					#### #update the biases
					sess.run([RBM.update_bias_h,RBM.update_bias_v],feed_dict={RBM.v:RBM.my_input_data})


					# increase the learnrate
					learnrate+=d_learnrate

					#### liveplot
					if RBM.liveplot and plt.fignum_exists(fig.number) and start%40==0:
						ax.cla()
						# ax[1].cla()
						# ax[2].cla()
						rbm_shape=int(sqrt(RBM.visible_units))
						matrix_new=tile_raster_images(X=RBM.w_i.T, img_shape=(rbm_shape, rbm_shape), tile_shape=(10, 10), tile_spacing=(0,0))
						ax.matshow(matrix_new)
						# ax[1].plot(errors)
						# ax[2].matshow(ubv.reshape(28,28))
						plt.pause(0.00001)
						
				log.info("Learnrate:",round(learnrate,4))
				log.info("Error",round(error_i,4))
				log.end() #ending the epoch


			log.end() #ending training the rbm 

	weights=[]
	for i in range(len(RBMs)):
		weights.append(RBMs[i].w.eval())


	#### Pre training is ended - create the DBM with the gained weights
	DBM=DBM_class(RBMs, [RBMs[0].w.eval(),RBMs[1].w.eval()])
	sess.run(tf.global_variables_initializer())

	if training:
		error_=[]
		for epoch in range(epochs):
			log.start("Deep BM Epoch:",epoch+1,"/",epochs)
			
			for start, end in zip( range(0, len(train_data), batchsize), range(batchsize, len(train_data), batchsize)):
				#### define a batch
				batch = train_data[start:end]

				sess.run([DBM.update_w1,DBM.update_w2],{DBM.v:batch})
				error_.append(DBM.error.eval({DBM.v:batch}))
			log.end()

	### testing the Deep BM
	for start, end in zip( range(0, len(train_data)/4, batchsize), range(batchsize, len(train_data)/4, batchsize)):
			#### define a batch
			batch = train_data[start:end]

			probs      = DBM.v_recon_prob.eval({DBM.v:batch})
			rec        = DBM.v_recon.eval({DBM.v:batch})
			test_error = DBM.error.eval({DBM.v:batch})


plt.plot(error_)
plt.show()



	#### Propagating the whole DBM forward and backward
	# h1       = DBM[0].h.eval({DBM[0].v:test_data[0:11]})
	# h2       = DBM[1].h.eval({DBM[1].v:h1})
	# v_prob3 = DBM[2].v_prob.eval({DBM[2].v:h2})
	# v_prob2 = DBM[1].v_prob_rev.eval({DBM[1].h_rev:v_prob3})
	# v_prob1 = DBM[0].v_prob_rev.eval({DBM[0].h_rev:v_prob2})

### End of Session


if pre_training:
	plt.close(fig)
	log.reset()
	log.info("Train Error:",error_i)
	# log.info("Test Error:",mean_test_error[0])
	log.info("Learnrate:",round(learnrate,4)," // Batchsize:",batchsize," // Temp.:",temp)


# Savin to file
if save_to_file:
	save(str(error_i),w_i,ubv,ubh)


####################################################################################################################################
#### Plot

	# Plot the Weights, Errors and other informations

if pre_training:
	for rbm in RBMs:		
		rbm.shape_o=int(sqrt(rbm.visible_units))
		weights_raster=tile_raster_images(X=rbm.w_i.T, img_shape=(rbm.shape_o, rbm.shape_o), tile_shape=(10, 20), tile_spacing=(0,0))
		map1=plt.matshow(rbm.w_i.T)
		plt.colorbar(map1)
		plt.matshow(weights_raster)

		# Plot the Test Phase	
		# if rbm.visible_units==784:
		# 	fig3,ax3=plt.subplots(2,10,figsize=(16,4))

		# 	for i in range(len(v_prob1)-1):
		# 		# plot the input
		# 		ax3[0][i].matshow(half_images[i:i+1].reshape(28,28))
		# 		# and the reconstruction
		# 		ax3[1][i].matshow(v_prob1[i:i+1].reshape(28,28))
				
fig3,ax3=plt.subplots(3,10,figsize=(16,4))
for i in range(10):
	# plot the input
	ax3[0][i].matshow(batch[i:i+1].reshape(28,28))
	# plot the probs of visible layer
	ax3[1][i].matshow(probs[i:i+1].reshape(28,28))
	# plot the recunstructed image
	ax3[2][i].matshow(rec[i:i+1].reshape(28,28))
	# plt.matshow(random_recon.reshape(28,28))



plt.show()
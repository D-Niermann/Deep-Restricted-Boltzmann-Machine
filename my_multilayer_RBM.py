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


################################################################################################################################################
### Class RBM 
class RBM(object):
	""" defines a 2 layer restricted boltzmann machine - first layer = input, second
	layer = output. Training with contrastive divergence """

	def __init__(self,vu,hu,forw_mult,back_mult,liveplot):
		#### User Variables

		self.hidden_units  = hu
		self.visible_units = vu


		self.liveplot       = liveplot

		self.forw_mult = forw_mult
		self.back_mult = back_mult

	# def graph(self):
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

####################################################################################################################################
#### User Settings ###
num_batches   = 500
epochs        = 1
learnrate     = 0.02
learnrate_max = 0.02
temp          = 1.0
training      = 1

save_to_file   = 0
load_from_file = 0
file_suffix    = ""

#### Create RBMs and merge them into one list for iteration###
#RBM(visible units, hidden units, forw_mult, back_mult, liveplot,...)
rbm1=RBM(784, 18*18, 2, 1, 0)
rbm2=RBM(18*18, 100, 2, 2, 0)
rbm3=RBM(100, 10, 1, 2, 0)
DBM=[rbm1,rbm2]

####################################################################################################################################
#### Session ####
time_now = time.asctime()
log.reset()
log.start("Session")
log.info(time_now)

errors         = []
mean_w_        = []
update         = 0
batchsize      = 55000/num_batches 
num_of_updates = epochs*num_batches
d_learnrate    = float(learnrate_max-learnrate)/num_of_updates


# define a figure for liveplotting
if training:
	fig,ax=plt.subplots(1,1,figsize=(15,10))

# start the session
with tf.Session() as sess:
	""" den code hier pro rbm oder global ausfueren? """
	sess.run(tf.global_variables_initializer())
	
	# if load_from_file:
	# 	sess.run(w.assign(init_pretrained(file_suffix,w=1)[0]))
	# 	sess.run(bias_v.assign(init_pretrained(file_suffix,bias_v=1)[0]))
	# 	sess.run(bias_h.assign(init_pretrained(file_suffix,bias_h=1)[0]))

	#iterate through the DBM , each iteration is a RBM
	for RBM_i,RBM in enumerate(DBM):

		if training:
			log.out("Training ",str(RBM_i+1)+".", "RBM")
			

			for epoch in range(epochs):
				log.start("Epoch:",epoch+1,"/",epochs)
				
				for start, end in zip( range(0, len(train_data), batchsize), range(batchsize, len(train_data), batchsize)):
					#### define a batch
					batch = train_data[start:end]

					my_input_data= batch
					if RBM_i>0:
						my_input_data=DBM[RBM_i-1].h.eval({DBM[RBM_i-1].v:batch})


					# output_1st_RBM=RBM1.v_recon(train_batch)
					""" klassen müssen noch richtig def sein """
					# output_2nd_RBM=RBM2.v_recon(output_1st_RBM)






					#### update the weights
					RBM.w_i,error_i=sess.run([RBM.update_w,RBM.error],feed_dict={RBM.v:my_input_data})

					#### #update the biases
					sess.run([RBM.update_bias_h,RBM.update_bias_v],feed_dict={RBM.v:my_input_data})

					# for plotting
					# errors.append(error_i)

					# increase the learnrate
					learnrate+=d_learnrate

					#### liveplot
					if RBM.liveplot and plt.fignum_exists(fig.number):
						ax.cla()
						# ax[1].cla()
						# ax[2].cla()

						matrix_new=tile_raster_images(X=RBM.w_i.T, img_shape=(28, 28), tile_shape=(10, 10), tile_spacing=(0,0))
						ax.matshow(matrix_new)
						# ax[1].plot(errors)
						# ax[2].matshow(ubv.reshape(28,28))
						plt.pause(0.00001)
						
				log.info("\tLearnrate:",round(learnrate,2))
				log.info("\tError",round(error_i,3))
				log.end() #ending the epoch

		#### Testing the network
		if RBM_i==0:
			half_images = test_data[0:11]
			#halfing some images from test_data
			half_images[1:6,500:] = 0
			RBM.probs      = RBM.v_prob.eval({RBM.v:half_images})
			RBM.rec        = RBM.v_recon.eval({RBM.v:half_images})

			mean_test_error = sess.run([RBM.error],feed_dict={RBM.v:test_data})


		#reverse feeding
		# input_vector=np.round(np.zeros([1,hidden_units]))
		# fig,ax2=plt.subplots(2,1,figsize=(10,10))
		# for i in range(100):
			# output=v_prob_rev.eval({h_rev:input_vector})
			# input_vector=h.eval({v:output})
			# ax2[0].cla()
			# ax2[1].cla()
			# ax2[0].matshow(output.reshape(28,28))
			# ax2[1].matshow(input_vector[:,:400].reshape(20,20))
			# plt.pause(0.0001)
### End of Session


if training:
	log.reset()
	log.info("Train Error:",error_i)
	log.info("Test Error:",mean_test_error[0])
	log.info("Learnrate:",round(learnrate)," // Batchsize:",batchsize," // Temp.:",temp)
log.end()

# Savin to file
if save_to_file:
	save(str(error_i),w_i,ubv,ubh)


####################################################################################################################################
#### Plot
for rbm in DBM:
	# Plot the Weights, Errors and other informations
	rbm.shape_o=int(sqrt(rbm.visible_units))
	if training:
		#plot the errors
		# plt.figure("Errors")
		# plt.plot(errors[10:])

		# plt.figure("Mean of W")
		# plt.plot(mean_w_)
		
		#plot the weights
		
		weights_raster=tile_raster_images(X=rbm.w_i.T, img_shape=(rbm.shape_o, rbm.shape_o), tile_shape=(10, 20), tile_spacing=(0,0))
		fig_m=plt.figure("Weights",figsize=(8,3))
		ax_m=fig_m.add_subplot(1,1,1)
		map1=ax_m.matshow(rbm.w_i.T)
		plt.colorbar(map1)
		plt.matshow(weights_raster)
		
		#plot the bias_v
		# map3=plt.matshow(ubv.reshape(rbm.shape_o,rbm.shape_o))
		# plt.colorbar(map3)

	# # Plot the Test Phase	
	# fig3,ax3=plt.subplots(3,10,figsize=(16,4))
	# if rbm.visible_units==784:
	# 	for i in range(len(rec)-1):
	# 		# plot the input
	# 		ax3[0][i].matshow(half_images[i:i+1].reshape(28,28))
	# 		# plot the probs of visible layer
	# 		ax3[1][i].matshow(rbm.probs[i:i+1].reshape(28,28))
	# 		# plot the recunstructed image
	# 		ax3[2][i].matshow(rbm.rec[i:i+1].reshape(28,28))
	# 		# plt.matshow(random_recon.reshape(rbm.shape_o,rbm.shape_o))



plt.show()
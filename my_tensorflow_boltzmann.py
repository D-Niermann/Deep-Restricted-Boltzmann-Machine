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


################################################################################################################################################
#### User Variables
writer_on     = False
hidden_units  = 200
visible_units = 784
batchsize     = 55000/500 # dividing by one will not work, at least 2 batches are required here
epochs        = 1
learnrate     = 0.25
temp          = 1.0

save_to_file   = 0
load_from_file = 0
training       = 1
liveplot       = 1



################################################################################################################################################
#### Graph
#### define each layer and the weight matrix w
v       = tf.placeholder(tf.float32,[None,visible_units],name="Visible-Layer")
w       = tf.Variable(tf.zeros([visible_units,hidden_units]),name="Weights")
bias_v  = tf.Variable(tf.zeros([visible_units]),name="Visible-Bias")
bias_h  = tf.Variable(tf.zeros([hidden_units]),name="Hidden-Bias")


# get the probabilities of the hidden units in 
h_prob  = sigmoid(tf.matmul(v,w) + bias_h,temp)
# get the actual activations for h {0,1}
h       = tf.nn.relu(
	            tf.sign(
	            	h_prob - tf.random_uniform(tf.shape(h_prob)) 
	            ) 
        	) 

# and the same for visible units
v_prob  = sigmoid(tf.matmul(h,tf.transpose(w)) + bias_v,temp)
v_recon = tf.nn.relu(
			tf.sign(
				v_prob - tf.random_uniform(tf.shape(v_prob))
				)
			)

# now get the probabilities of h again from the reconstructed v_recon
h_gibbs = sigmoid(tf.matmul(v_recon, w) + bias_h,temp) 

##### define reconstruction error and the energy  
# energy = -tf.reduce_sum(bias_v*v_recon)-tf.reduce_sum(bias_h*h)-tf.matmul(tf.matmul(h,tf.transpose(w)), v_recon)
error  = tf.reduce_mean(tf.square(v-v_recon))

#### Training with Contrastive Divergence
pos_grad  = tf.matmul(tf.transpose(v),h)
neg_grad  = tf.matmul(tf.transpose(v_recon),h_gibbs)
numpoints = tf.cast(tf.shape(v)[0],tf.float32) #number of train inputs per batch (for averaging the CD matrix??)
# weight update
CD       = (pos_grad - neg_grad)/numpoints
update_w = w.assign(w+learnrate*CD)
mean_w   = tf.reduce_mean(w)
#update bias
""" warum hier reduce_mean(...,0)?? -> das gibt sogar einen vector mit shape (784,)"""
update_bias_v = bias_v.assign(bias_v+learnrate*tf.reduce_mean(v-v_recon,0))
update_bias_h = bias_h.assign(bias_h+learnrate*tf.reduce_mean(h-h_gibbs,0))


####################################################################################################################################
#### Session ####
log.reset()
time_now = time.asctime()
log.start("Session")
log.info(time_now)

errors=[]
mean_w_=[]

# define a figure for liveplotting
if training and liveplot:
	fig,ax=plt.subplots(1,1,figsize=(15,10))

# start the session
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	if load_from_file:
		sess.run(w.assign(init_pretrained(w=1)[0]))
		sess.run(bias_v.assign(init_pretrained(bias_v=1)[0]))
		sess.run(bias_h.assign(init_pretrained(bias_h=1)[0]))

	if training:
		for epoch in range(epochs):
			log.start("epoch:",epoch)
			
			for start, end in zip( range(0, len(train_data), batchsize), range(batchsize, len(train_data), batchsize)):
				#### define a batch
				batch = train_data[start:end]

			
				#### update the weights
				w_i,error_i=sess.run([update_w,error],feed_dict={v:batch})

				#### #update the biases
				ubh,ubv=sess.run([update_bias_h,update_bias_v],feed_dict={v:batch})

				# mean_w_.append(mean_w.eval())
				errors.append(error_i)


				#### plot
				if liveplot:
					ax.cla()
					# ax[1].cla()
					# ax[2].cla()
					
					matrix_new=tile_raster_images(X=w_i.T, img_shape=(28, 28), tile_shape=(10, 10), tile_spacing=(0,0))
					ax.matshow(matrix_new)
					# ax[1].plot(errors)
					# ax[2].matshow(ubv.reshape(28,28))
					plt.pause(0.00001)
			
			log.info("\tError",error_i)
			log.end() #ending the epoch

	#### Testing the network
	probs=v_prob.eval({v:train_data[0:9]})
	rec=v_recon.eval({v:train_data[0:9]})

	random_recon=v_recon.eval({v:rnd.random([1,784])})

if training:
	log.reset()
	log.info("Error:",error_i)
	log.info("Learnrate:",learnrate," // Batchsize:",batchsize," // Temp.:",temp)
log.end()


####################################################################################################################################
#### Plot

# Plot the Weights, Errors and other informations
if training:
	#plot the errors
	plt.figure("Errors")
	plt.plot(errors)
	# plt.figure("Mean of W")
	# plt.plot(mean_w_)
	#plot the weights
	weights_raster=tile_raster_images(X=w_i.T, img_shape=(28, 28), tile_shape=(10, 20), tile_spacing=(0,0))
	map1=plt.matshow(w_i)
	plt.colorbar(map1)
	plt.matshow(weights_raster)
	#plot the bias_v
	map3=plt.matshow(ubv.reshape(28,28))
	plt.colorbar(map3)

# Plot the Test Phase	
fig3,ax3=plt.subplots(3,8,figsize=(16,4))
for i in range(len(rec)-1):
	# plot the input
	ax3[0][i].matshow(train_data[i:i+1].reshape(28,28))
	# plot the probs of visible layer
	ax3[1][i].matshow(probs[i:i+1].reshape(28,28))
	# plot the recunstructed image
	ax3[2][i].matshow(rec[i:i+1].reshape(28,28))
plt.matshow(random_recon.reshape(28,28))
plt.show()

# Savin to file
if save_to_file:
	save(str(error_i),w_i,ubv,ubh)
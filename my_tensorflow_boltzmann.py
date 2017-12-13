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
hidden_units  = 500
visible_units = 784


num_batches   = 700
epochs        = 4
learnrate     = 0.03
learnrate_max = 1.
temp          = 2.5 

save_to_file   = 0
load_from_file = 0

training       = 1
liveplot       = 1



################################################################################################################################################
#### Graph
#### define each layer and the weight matrix w
""" shape definitions are wired here-normally one would define [rows,columns] but here it is reversed with [columns,rows]?"""

v       = tf.placeholder(tf.float32,[None,visible_units],name="Visible-Layer") # has shape [number of images per batch,number of visible units]

w       = tf.Variable(tf.random_uniform([visible_units,hidden_units],minval=-1e-3,maxval=1e-3),name="Weights")# init with small random values to break symmetriy
bias_v  = tf.Variable(tf.zeros([visible_units]),name="Visible-Bias")
bias_h  = tf.Variable(tf.zeros([hidden_units]),name="Hidden-Bias")


# get the probabilities of the hidden units in 
h_prob  = sigmoid(tf.matmul(v,w) + bias_h,temp)
#h has shape [number of images per batch, number of hidden units]
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
#matrix shape is untouched throu the batches because w*v=h even if v has more columns, but dividing be numpoints is recomended since CD
# [] = [784,batchsize]-transposed v * [batchsize,500] -> [784,500] - like w 
pos_grad  = tf.matmul(tf.transpose(v),h)
eval_pos  = tf.shape(pos_grad)
neg_grad  = tf.matmul(tf.transpose(v_recon),h_gibbs)
numpoints = tf.cast(tf.shape(v)[0],tf.float32) #number of train inputs per batch (for averaging the CD matrix -> see practical paper by hinton)
# weight update
CD       = (pos_grad - neg_grad)/numpoints
update_w = w.assign(w+learnrate*CD)
mean_w   = tf.reduce_mean(w)
#update bias
""" Since vectors v and h are actualy matrices with number of batch_size images in them, reduce mean will make them to a vector again """
update_bias_v = bias_v.assign(bias_v+learnrate*tf.reduce_mean(v-v_recon,0))
update_bias_h = bias_h.assign(bias_h+learnrate*tf.reduce_mean(h-h_gibbs,0))

# reverse feed
h_rev       = tf.placeholder(tf.float32,[None,hidden_units],name="Reverse-hidden")
v_prob_rev  = sigmoid(tf.matmul(h_rev,tf.transpose(w)) + bias_v,temp)
v_recon_rev = tf.nn.relu(tf.sign(v_prob - tf.random_uniform(tf.shape(v_prob))))

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
				sess.run([update_bias_h,update_bias_v],feed_dict={v:batch})

				# mean_w_.append(mean_w.eval())
				errors.append(error_i)

				# increase the learnrate
				learnrate+=d_learnrate

				#### plot
				if liveplot and plt.fignum_exists(fig.number):
					ax.cla()
					# ax[1].cla()
					# ax[2].cla()

					matrix_new=tile_raster_images(X=w_i.T, img_shape=(28, 28), tile_shape=(10, 10), tile_spacing=(0,0))
					ax.matshow(matrix_new)
					# ax[1].plot(errors)
					# ax[2].matshow(ubv.reshape(28,28))
					plt.pause(0.00001)
					
			log.info("\tLearnrate:",round(learnrate,2))
			log.info("\tError",round(error_i,3))
			log.end() #ending the epoch

	#### Testing the network
	half_images=train_data[0:9]
	half_images[1:6,500:]=0
	probs=v_prob.eval({v:half_images})
	rec=v_recon.eval({v:half_images})

	#reverse feeding
	input_vector=np.round(np.zeros([1,hidden_units]))
	# fig,ax2=plt.subplots(2,1,figsize=(10,10))
	for i in range(100):
		output=v_prob_rev.eval({h_rev:input_vector})
		input_vector=h.eval({v:output})
		# ax2[0].cla()
		# ax2[1].cla()
		# ax2[0].matshow(output.reshape(28,28))
		# ax2[1].matshow(input_vector[:,:400].reshape(20,20))
		# plt.pause(0.0001)

if training:
	log.reset()
	log.info("Error:",error_i)
	log.info("Learnrate:",round(learnrate)," // Batchsize:",batchsize," // Temp.:",temp)
log.end()


####################################################################################################################################
#### Plot

# Plot the Weights, Errors and other informations
if training:
	#plot the errors
	plt.figure("Errors")
	plt.plot(errors[10:])
	# plt.figure("Mean of W")
	# plt.plot(mean_w_)
	#plot the weights
	weights_raster=tile_raster_images(X=w_i.T, img_shape=(28, 28), tile_shape=(10, 20), tile_spacing=(0,0))
	fig_m=plt.figure("Weights",figsize=(8,3))
	ax_m=fig_m.add_subplot(1,1,1)
	map1=ax_m.matshow(w_i.T)
	plt.colorbar(map1)
	plt.matshow(weights_raster)
	#plot the bias_v
	# map3=plt.matshow(ubv.reshape(28,28))
	# plt.colorbar(map3)

# Plot the Test Phase	
fig3,ax3=plt.subplots(3,8,figsize=(16,4))
for i in range(len(rec)-1):
	# plot the input
	ax3[0][i].matshow(train_data[i:i+1].reshape(28,28))
	# plot the probs of visible layer
	ax3[1][i].matshow(probs[i:i+1].reshape(28,28))
	# plot the recunstructed image
	ax3[2][i].matshow(rec[i:i+1].reshape(28,28))
# plt.matshow(random_recon.reshape(28,28))

#plot the reverse outputs
# plt.matshow(output.reshape(28,28))


# Savin to file
if save_to_file:
	save(str(error_i),w_i,ubv,ubh)

plt.show()
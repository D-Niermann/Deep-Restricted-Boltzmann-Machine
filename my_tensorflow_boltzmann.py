#### Imports
if True:
	# -*- coding: utf-8 -*-
	print "Starting..."
	import numpy as np
	import numpy.random as rnd
	import matplotlib.pyplot as plt
	import matplotlib.image as img
	# import scipy.ndimage.filters as filters
	# import pandas as pd
	import os,time,sys
	from math import exp,sqrt,sin,pi,cos,log
	np.set_printoptions(precision=3)
	# plt.style.use('ggplot')
	os.chdir("/Users/Niermann/Google Drive/Masterarbeit/Python")
	from Logger import *
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
	import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#### Load MNIST Data 
if "train_data" not in globals():
	log.out("Loading Data")
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	train_data, trY, test_data, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

#### Functions

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),scale_rows_to_unit_interval=True,output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array

def save(w=[],bias_v=[],bias_h=[]):
	path="/Users/Niermann/Google Drive/Masterarbeit/Python"
	os.chdir(path)
	if len(w)!=0:
		np.savetxt("weights.txt", w)
	if len(bias_v)!=0:
		np.savetxt("bias_v.txt", bias_v)
	if len(bias_h)!=0:
		np.savetxt("bias_h.txt", bias_h)
	print "saved weights and biases"

def init_pretrained(w=None,bias_v=None,bias_h=None):
	path="/Users/Niermann/Google Drive/Masterarbeit/Python"
	os.chdir(path)
	m=[]
	if (w)!=None:
		w=np.loadtxt("weights.txt")
		m.append(w)
	if (bias_v)!=None:
		bias_v=np.loadtxt("bias_v.txt")
		m.append(bias_v)
	if (bias_h)!=None:
		bias_h=np.loadtxt("bias_h.txt")
		m.append(bias_h)
	print "loaded %s objects from file"%str(len(m))
	return m


################################################################################################################################################
#### User Variables
writer_on      = False
hidden_units   = 500
visible_units  = 784
batchsize      = 55000/100 # dividing by one will not work, at least 2 batches are required here
epochs         = 1
learnrate      = 1.
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
h_prob  = tf.nn.sigmoid(tf.matmul(v,w) + bias_h)
# get the actual activations for h {0,1}
h       = tf.nn.relu(
	            tf.sign(
	            	h_prob - tf.random_uniform(tf.shape(h_prob)) 
	            ) 
        	) 

# and the same for visible units
v_prob  = tf.nn.sigmoid(tf.matmul(h,tf.transpose(w)) + bias_v)
v_recon = tf.nn.relu(
			tf.sign(
				v_prob - tf.random_uniform(tf.shape(v_prob))
				)
			)

# now get the probabilities of h again from the reconstructed v_recon
h_gibbs = tf.nn.sigmoid(tf.matmul(v_recon, w) + bias_h) 

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
#update bias
""" warum hier reduce_mean(...,0)?? -> das gibt sogar einen vector mit shape (784,)"""
update_bias_v = bias_v.assign(bias_v+learnrate*tf.reduce_mean(v-v_recon,0))
update_bias_h = bias_h.assign(bias_h+learnrate*tf.reduce_mean(h-h_gibbs,0))

####################################################################################################################################
#### Session ####
time_now = time.asctime()
log.start("Session")
log.info(time_now)

errors=[]


# define a figure for liveplotting
if training and liveplot:
	fig,ax=plt.subplots(1,1,figsize=(15,10))

# start the session
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	if load_from_file:
		sess.run(w.assign(init_pretrained(w=1)[0]))

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
			
			print numpoints.eval({v:batch})
			log.info("\tError",error_i)
			log.end() #ending the epoch

	#### Testing the network
	probs=v_prob.eval({v:train_data[0:9]})
	rec=v_recon.eval({v:train_data[0:9]})

if training:
	log.info("Error:",error_i)
	log.info("Learnrate:",learnrate,"// Batchsize:",batchsize)
log.end()


####################################################################################################################################
#### Plot

# Plot the Weights, Errors and other informations
if training:
	#plot the errors
	plt.plot(errors)
	#plot the weights
	weights_raster=tile_raster_images(X=w_i.T, img_shape=(28, 28), tile_shape=(20, 20), tile_spacing=(0,0))
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

plt.show()

# Savin to file
if save_to_file:
	save(w_i)
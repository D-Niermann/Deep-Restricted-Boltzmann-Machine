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
	os.chdir("/Users/Niermann/Google Drive/Masterarbeit")
	### import seaborn? ###
	if 1:
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
	plt.rcParams['image.cmap'] = 'coolwarm'

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


def reshape_image(image):
	# makes a quadratic image to a vector
	return image.reshape(image.shape[0]*image.shape[1])

def sign(x,T):
	return 1./(1+np.exp(-1/T*x))


def stochastic_fire(vector):
	# sampling of the probability given by :vector:
	rng=rnd.random(len(vector))
	a=vector>rng
	return a.astype(int)


def forward_pass(v_input):
	# calculates hidden layer by given visible layer
	h=sign(bias_h+np.dot(w,v_input),temp)
	return h

def backward_pass(h_input):
	#caöculates visible layer 
	v=sign(bias_v+np.dot(w.T,h_input),temp)
	return v

def energy():
	#calculates energy of the whole system
	return -sum(bias_v*v)-sum(bias_h*h)-np.dot(np.dot(w.T, h),v)

def load_data(path,max_pics):
	# loads data from files and subfolders
	# :path: main folder
	# :max_pics: only takes this many pictures from each folder
	os.chdir(path)
	data=[]
	save=max_pics
	for folders in os.listdir(os.getcwd()):
		if folders[0]!="." and folders!="tmp":
			print "loading from: "+"..."+path[-20:]+folders
			os.chdir(path+folders)
			max_pics+=save
			print "Collected images:",str(len(data))
			for files in os.listdir(path+folders):
				if files[-4:]==".png":
					data.append(img.imread(files))
				if len(data)>=max_pics:
					break
	print "loaded data with %s mb"%str(sys.getsizeof(data)/1024.)
	return np.array(data)

def train(w):
	global v,h,image,numpoints,_v,bias_v,bias_h

	### set up v and h ###
	h=stochastic_fire(forward_pass(v))
	_v=backward_pass(h)
	v=stochastic_fire(_v)

	#positive gradient
	pos_grad=np.outer(h, image)
	
	# this is the fire prob of h (gibbs sampling)
	h_prob=forward_pass(v) 

	#negative gradient
	neg_grad=np.outer(h_prob, v)

	# contrastive divergence
	
	CD=(pos_grad-neg_grad)/numpoints

	#update biases
	bias_v+=learnrate*((image-v))
	bias_h+=learnrate*((h-h_prob))

	return CD

def init_pretrained(w=None,bias_v=None,bias_h=None):
	path="/Users/Niermann/Google Drive/Masterarbeit"
	os.chdir(path)
	m=[]
	if w!=None:
		w=np.loadtxt("weights.txt")
		m.append(w)
	if bias_v!=None:
		bias_v=np.loadtxt("bias_v.txt")
		m.append(bias_v)
	if bias_h!=None:
		bias_h=np.loadtxt("bias_h.txt")
		m.append(bias_h)
	print "loaded %s objects from file"%str(len(m))
	return m
	

def save(w,bias_v,bias_h):
	path="/Users/Niermann/Google Drive/Masterarbeit"
	os.chdir(path)
	np.savetxt("weights.txt", w)
	np.savetxt("bias_v.txt", bias_v)
	np.savetxt("bias_h.txt", bias_h)
	print "saved weights and biases"

def test_reconstruction(input_image):
	""" MALICIOUS"""
	global w,v,h

	f,ax=plt.subplots(1,2,figsize=(10,6))
	
	v=reshape_image(input_image)
	ax[0].matshow((input_image) ,label="image")
	ax[0].set_title("input image",y=1.1)

	### set up v and h ###
	h=stochastic_fire(forward_pass(v))
	v=(backward_pass(h))

	#plot v reconstructed
	mappable=ax[1].matshow(v.reshape(28,28),label="v",cmap="coolwarm")
	ax[1].set_title("visible layer",y=1.1)
	plt.colorbar(mappable)

	plt.show()

def liveplot(lp,interval,i):
	global ax1
	### plot ###
	if lp:
		if i%10==0:
			ax1[0].cla()
			ax1[1].cla()
			# ax1[2].clear()
			ax1[2].clear()
			ax1[3].clear()
			ax1[0].matshow(image)
			ax1[1].matshow(_v.reshape(28,28))	
			ax1[2].plot(energy_[10:],"b")
			# ax1[2].matshow(w)
			ax1[3].plot(error_[10:])

			plt.pause(interval)

########################################################################
########################################################################


### init all variables ###
v               = np.ones(28*28) #visible layer
h               = np.zeros(500) #hidden layer
# w,bias_v,bias_h = init_pretrained(1,1,1)
w             = rnd.random([h.size,v.size])/100000. #weights
bias_v        = np.zeros(v.size) #bias of visible layer
bias_h        = np.zeros(h.size) #bias of hidden layer
temp            = float(3) # 0.001 for heavyside function , 1 for completely random
end_temp        = 0.1 #temperature at end of epoch
learnrate       = 1.
epochs          = 1  # number of training steps“
lp              = False #if plotting while traingin is on
files_to_load   = 1000 #files per type of class
batch_size	    = 100 #how many images will be batched and averaged
########################################################################
########################################################################

if lp:
	fig1,ax1=plt.subplots(4,1,figsize=(5,10))
	n=4 ## number of subplots
	ax1[0].set_title("hidden layer",fontdict= {"y":5},loc="left")
	ax1[1].set_title("visible layer",fontdict={"y":5},loc="left")
	# ax1[2].set_title("Energy")
	ax1[2].set_title("Weights")

#######################################################################

energy_=[]
error_=[]
p_=[]
CD_=[]
m=0
switch=0
start_temp=temp

path_to_traindata = "/Users/Niermann/Downloads/MNIST/training/"
# train_data              = load_data(path_to_traindata, files_to_load)
# rnd.shuffle(train_data)
len_data         = float(len(train_data))
numpoints        = train_data[0].shape[0]*train_data[0].shape[0]
temp_step        = (start_temp-end_temp)/(len_data*epochs)
batch_start      = 0
batch_end        = batch_size
batches          = len_data/batch_size
if int(batches) != batches:
	raise TypeError("Error with batch size")
batches=int(batches)



t1=time.clock()
for i in range(epochs):
	
	print "-------------------"
	print "epoch: ",i

	for batch in range(batches):
		CD_=np.zeros([batches,w.shape[0],w.shape[1]])
		# print "batch",batch_start,"-",batch_end
	
		for j,image in enumerate(train_data[batch_start:batch_end]):
			#vectorize image 
			# image_vector=reshape_image(image)

			### set the input ###
			my_input = image
			v        = my_input

			#training (append CD matrix to list and update bias v and bias h online)
			CD = train(w)
			

			# append variables to lists
			CD_[batch,:,:]=CD
			energy_.append(energy())
			error_.append((np.mean(image-v)**2)*1000)

			#anneal the temp
			temp-=temp_step

			# printing and stuff
		
			# print "Finished to ",str(round(j/len_data,2)),"%"
			if switch==0 and j/len_data>=0.001 :
				switch=1
				t2=time.clock()
				percent=float(j)/len_data
				time_took=t2-t1
				est_time=time_took*epochs*(100/percent)/100
				print "est. time for training: ",round(est_time/60.,3),"min"

			#plotting
			if lp:
				liveplot(lp,0.0001,j)

		#update Weights
		w+=learnrate*np.mean(CD_,0)
		batch_start += batch_size
		batch_end   += batch_size

t3=time.clock()
print "training took: ",str(round(t3-t1,3)/60.),"min"
print "Mean Weights and decays:", np.round([np.mean(w),np.mean(bias_h),np.mean(bias_v)],2)
print "Mean Error: ",round(np.mean(error_),6), "temp: ",temp,"learnrate: ",learnrate
if lp:
	plt.close()
########################################################################################################
matrix_new=tile_raster_images(X=w, img_shape=(28, 28), tile_shape=(25, 20), tile_spacing=(1,1))
plt.matshow(matrix_new)


# save(w, bias_v, bias_h)



plt.show()
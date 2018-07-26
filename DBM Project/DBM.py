# -*- coding: utf-8 -*-
#### Imports
if True:
	# -*- coding: utf-8 -*-
	print ("Starting")
	
	import matplotlib as mpl
	import os,time,sys

	try: # if on macbook
		workdir="/Users/Niermann/Google Drive/Masterarbeit/Python/DBM Project"
		os.chdir(workdir)
		import_seaborn = True
	except: #else on university machine
		workdir="/home/dario/Dokumente/DBM Project"
		os.chdir(workdir)
		mpl.use("Agg") #use this to not display plots but save them
		import_seaborn = True

	data_dir=workdir+"/data"

	import numpy as np
	import numpy.random as rnd
	import matplotlib.pyplot as plt
	import tensorflow as tf
	from pandas import DataFrame,Series,read_csv
	
	from math import exp,sqrt,sin,pi,cos
	np.set_printoptions(precision=3)

		
	from Logger import *
	from RBM_Functions import *
	### import seaborn? ###
	if import_seaborn:
		import seaborn

		seaborn.set(font_scale=1.2)
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

	mpl.rcParams["image.cmap"] = "gray"
	mpl.rcParams["grid.linewidth"] = 0.5
	mpl.rcParams["lines.linewidth"] = 1.25
	mpl.rcParams["font.family"]= "serif"

		# plt.rcParams['image.cmap'] = 'coolwarm'
		# seaborn.set_palette(seaborn.color_palette("Set2", 10))

	log=Logger(True)

from tensorflow.examples.tutorials.mnist import input_data
time_now = time.asctime()
time_now = time_now.replace(":", "-")
time_now = time_now.replace(" ", "_")

#### Load T Data 
LOAD_MNIST = 1
LOAD_HORSES = 0
if "train_data" not in globals():
	if LOAD_MNIST:
		log.out("Loading Data")
		mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
		train_data, train_label, test_data, test_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
		

		#get test data of only one number class:
		index_for_number_test  = np.zeros([10,1200])
		where = np.zeros(10).astype(np.int)
		index_for_number_train = []
		for i in range(len(test_label)):
			for digit in range(10):
				d_array = np.zeros(10)
				d_array[digit] = 1
				if (test_label[i]==d_array).sum()==10:
					index_for_number_test[digit][where[digit]]=i
					where[digit]+=1
		index_for_number_test = index_for_number_test.astype(np.int)

		for i in range(len(train_label)):
			if (train_label[i]==[0,0,0,1,0,0,0,0,0,0]).sum()==10:
				index_for_number_train.append(i)



			# 	new_data_train = np.zeros([len(train_label),100])
			# 	new_data_test = np.zeros([len(test_label),100])
			# 	for i in range(len(train_label)):
			# 		new_data_train[i]  = np.repeat(train_label[i],10).reshape(10,10).T.flatten()
			# 	for i in range(len(test_label)):
			# 		new_data_test[i]  = np.repeat(test_label[i],10).reshape(10,10).T.flatten()
			# 	train_label = new_data_train
			# 	test_label = new_data_test
			# log.out("Setting trian label = 10*10")
				# test_data_noise = np.copy(test_data)
				# # making noise 
				# for i in range(len(test_data_noise)):
				# 	test_data_noise[i]  += np.round(rnd.random(test_data_noise[i,:].shape)*0.55)
				# 	# half_images[i] = abs(half_images[i])
				# 	# half_images[i] *= 1/half_images[i].max()
				# 	# half_images[i] *= rnd.random(half_images[i].shape)
				# test_data_noise   = test_data_noise>0

				# noise_data_train = sample_np(rnd.random(train_data.shape)*0.2)
				# noise_data_test = sample_np(rnd.random(test_data.shape)*0.2)
				# noise_label_train = np.zeros(train_label.shape)
				# noise_label_test = np.zeros(test_label.shape)

	if LOAD_HORSES:
		log.out("Loading HORSE Data")
		horse_data_dir   = workdir+"/Horse_data_rescaled/"
		files      = os.listdir(horse_data_dir)
		train_data = np.zeros([len(files)-50,64**2])
		test_data  = np.zeros([50,64**2])

		from PIL import Image
		for i,f in enumerate(files):
			if f[-4:]==".jpg":
				img_data = np.array(Image.open(horse_data_dir+f)).flatten()/255.
				if i < train_data.shape[0]:
					train_data[i] = img_data
				else:
					test_data[i-train_data.shape[0]] = img_data
			else:
				log.info("Skipped %s"%f)

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
		#shape definitions are wired here-normally one would define [rows,columns] but here it is reversed with [columns,rows]? """

		self.v       = tf.placeholder(tf.float32,[None,self.visible_units],name="Visible-Layer") 
		# has shape [number of images per batch,number of visible units]

		self.w       = tf.Variable(tf.random_normal([self.visible_units,self.hidden_units], stddev=0.00001),name="Weights")
		self.bias_v  = tf.Variable(tf.zeros([self.visible_units]),name="Visible-Bias")
		self.bias_h  = tf.Variable(tf.zeros([self.hidden_units]), name="Hidden-Bias")


		# get the probabilities of the hidden units in w
		self.h_prob  = sigmoid(tf.matmul(self.v,self.forw_mult*self.w) + self.bias_h,temp)
		# h has shape [number of images per batch, number of hidden units]
		# get the actual activations for h {0,1}
		# self.h       = tf.nn.relu(
		# 	            tf.sign(
		# 	            	self.h_prob - tf.random_uniform(tf.shape(self.h_prob)) 
		# 	            	) 
		#         		) 

		# and the same for visible units
		self.v_prob  = sigmoid(tf.matmul(self.h_prob,(self.back_mult*self.w),transpose_b=True) + self.bias_v,temp)
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
		self.pos_grad  = tf.matmul(self.v,self.h_prob,transpose_a=True)
		self.neg_grad  = tf.matmul(self.v_recon,self.h_gibbs,transpose_a=True)
		self.numpoints = tf.cast(tf.shape(self.v)[0],tf.float32) 
		#number of train inputs per batch (for averaging the CD matrix -> see practical paper by hinton)
		# contrastive divergence
		self.CD        = (self.pos_grad - self.neg_grad)/self.numpoints
		

		#update w
		self.update_w = self.w.assign_add(self.learnrate*self.CD)
		self.mean_w   = tf.reduce_mean(self.w)

		#update bias
		""" Since vectors v and h are actualy matrices with number of batch_size images in them, reduce mean will make them to a vector again """
		self.update_bias_v = self.bias_v.assign_add(self.learnrate*tf.reduce_mean(self.v-self.v_recon,0))
		self.update_bias_h = self.bias_h.assign_add(self.learnrate*tf.reduce_mean(self.h_prob-self.h_gibbs,0))


		# reverse feed
		# self.h_rev       = tf.placeholder(tf.float32,[None,self.hidden_units],name="Reverse-hidden")
		# self.v_prob_rev  = sigmoid(tf.matmul(self.h_rev,(self.w),transpose_b=True) + self.bias_v,temp)
		# self.v_recon_rev = tf.nn.relu(tf.sign(self.v_prob_rev - tf.random_uniform(tf.shape(self.v_prob_rev))))

	def train(self,sess,RBM_i,RBMs,batch):
		self.my_input_data = batch
		# iterate which RBM level this is and calculate the proper input 
		for j in range(1,len(RBMs)):
			if RBM_i >= j:
				self.my_input_data = RBMs[j-1].h_prob.eval({RBMs[j-1].v : self.my_input_data})

		#### update the weights and biases
		self.w_i, self.error_i = sess.run([self.update_w,self.error],feed_dict={self.v:self.my_input_data})
		sess.run([self.update_bias_h,self.update_bias_v],feed_dict={self.v:self.my_input_data})

		return self.w_i,self.error_i




################################################################################################################################################
### Class Deep BM 
class DBM_class(object):
	"""defines a deep boltzmann machine
	"""
	def __init__(self,shape,liveplot,classification):
		self.n_layers       = len(shape)
		self.liveplot       = liveplot # if true will open a lifeplot of the weight matrix 
		self.SHAPE          = shape  # contains the number of  neurons in a list from v layer to h1 to h2 
		self.classification = classification #weather the machine uses a label layer 
		
		self.init_state     = 0
		self.exported       = 0
		self.tested         = 0
		self.l_mean         = np.zeros([self.n_layers])
		
		# self.n_particles = 0; # not in use

		self.train_time            = 0	# counts train time in seconds
		self.epochs                = 0	# epoch counter
		self.update                = 0 	# update counter
		self.recon_error_train     = np.zeros([N_BATCHES_TRAIN*N_EPOCHS_TRAIN/10, self.n_layers])	# save reconstructon error for every batch
		self.class_error_train     = np.zeros([N_BATCHES_TRAIN*N_EPOCHS_TRAIN/10, self.n_layers])	# -"- train error -"-
		self.layer_diversity_train = np.zeros([N_BATCHES_TRAIN*N_EPOCHS_TRAIN/10, self.n_layers])	# save layer variance across batch for every batch in train function
		self.layer_act_train       = np.zeros([N_BATCHES_TRAIN*N_EPOCHS_TRAIN/10, self.n_layers])	# save how many units are active across one layer in % for every batch
		
		self.freerun_diff_train    = np.zeros([N_BATCHES_TRAIN*N_EPOCHS_TRAIN/10, self.n_layers])


		### save dictionary where time series data from test and train is stored
		self.save_dict ={	"Test_Epoch":    [],
							"Train_Epoch":   [],
							"Recon_Error":   [],
							"Class_Error":   [],
							"Class_Error_Train_Data":   [],
							"Temperature":   [],
							"Learnrate":     [],
							"Freerun_Steps": [],
							}
		for i in range(self.n_layers-1):
			self.save_dict["W_mean_%i"%i] = []
			self.save_dict["W_diff_%i"%i] = []
			self.save_dict["CD_abs_mean_%i"%i] = []
		for i in range(self.n_layers):
			self.save_dict["Bias_diff_%i"%i] = []
			self.save_dict["Layer_Activity_%i"%i] = []
			self.save_dict["Layer_Diversity_%i"%i] = []

		### log list where all constants are saved
		self.log_list =	[["SHAPE",                self.SHAPE],
						["N_EPOCHS_PRETRAIN",     N_EPOCHS_PRETRAIN], 
						["N_BATCHES_PRETRAIN",    N_BATCHES_PRETRAIN], 
						["N_BATCHES_TRAIN",       N_BATCHES_TRAIN], 
						["LEARNRATE_PRETRAIN",    LEARNRATE_PRETRAIN], 
						["LEARNRATE_START",       LEARNRATE_START], 
						["LEARNRATE_SLOPE",       LEARNRATE_SLOPE], 
						["TEMP_START",            TEMP_START], 
						["TEMP_SLOPE",            TEMP_SLOPE], 
						["PATHSUFFIX_PRETRAINED", PATHSUFFIX_PRETRAINED], 
						["PATHSUFFIX",            PATHSUFFIX], 
						["DO_LOAD_FROM_FILE",      DO_LOAD_FROM_FILE], 
						["TEST_EVERY_EPOCH",      TEST_EVERY_EPOCH],
						["USE_DROPOUT",			  USE_DROPOUT],
						["DROPOUT_RATE", 		  DROPOUT_RATE],
						["DO_NORM_W", 			  DO_NORM_W],
						]## append variables that change during training in the write_to_file function


		log.out("Creating RBMs")
		self.RBMs    = [None]*(self.n_layers-1)
		for i in range(len(self.RBMs)):
			if i == 0 and len(self.RBMs)>1:
				self.RBMs[i] = RBM(self.SHAPE[i],self.SHAPE[i+1], forw_mult= 1, back_mult = 1, learnrate = LEARNRATE_PRETRAIN, liveplot=0)
				log.out("2,1")
			elif i==len(self.RBMs)-1 and len(self.RBMs)>1:
				self.RBMs[i] = RBM(self.SHAPE[i],self.SHAPE[i+1], forw_mult= 1, back_mult = 1, learnrate = LEARNRATE_PRETRAIN, liveplot=0)				
				log.out("1,2")
			else:
				if len(self.RBMs) == 1:
					self.RBMs[i] = RBM(self.SHAPE[i],self.SHAPE[i+1], forw_mult= 1, back_mult = 1, learnrate = LEARNRATE_PRETRAIN, liveplot=0)
					log.out("1,1")
				else:
					self.RBMs[i] = RBM(self.SHAPE[i],self.SHAPE[i+1], forw_mult= 1, back_mult = 1, learnrate = LEARNRATE_PRETRAIN, liveplot=0)
					log.out("2,2")
				
		# self.RBMs[1] = RBM(self.SHAPE[1],self.SHAPE[2], forw_mult= 1, back_mult = 1, learnrate = LEARNRATE_PRETRAIN, liveplot=0)
		# self.RBMs[2] = RBM(self.SHAPE[2],self.SHAPE[3], forw_mult= 1, back_mult = 1, learnrate = LEARNRATE_PRETRAIN, liveplot=0)

	def pretrain(self):
		""" this function will pretrain the RBMs and define a self.weights list where every
		weight will be stored in. This weights list can then be used to save to file and/or 
		to be loaded into the DBM for further training. 
		"""

		if DO_PRETRAINING:
			for rbm in self.RBMs:
				if rbm.liveplot:
					log.info("Liveplot is open!")
					fig,ax=plt.subplots(1,1,figsize=(15,10))
					break

		batchsize_pretrain = int(len(train_data)/N_BATCHES_PRETRAIN)

		with tf.Session() as sess:
			# train session - v has batchsize length
			log.start("Pretrain Session")
			
			
			#iterate through the RBMs , each iteration is a RBM
			if DO_PRETRAINING:	
				sess.run(tf.global_variables_initializer())

				for RBM_i, RBM in enumerate(self.RBMs):
					log.start("Pretraining ",str(RBM_i+1)+".", "RBM")
					

					for epoch in range(N_EPOCHS_PRETRAIN[RBM_i]):

						log.start("Epoch:",epoch+1,"/",N_EPOCHS_PRETRAIN[RBM_i])
						
						for start, end in zip( range(0, len(train_data), batchsize_pretrain), range(batchsize_pretrain, len(train_data), batchsize_pretrain)):
							#### define a batch
							batch = train_data[start:end]
							# train the rbm  
							w_i,error_i = RBM.train(sess,RBM_i,self.RBMs,batch)
							#### liveplot
							if RBM.liveplot and plt.fignum_exists(fig.number) and start%40==0:
								ax.cla()
								rbm_shape  = int(sqrt(RBM.visible_units))
								matrix_new = tile_raster_images(X=w_i.T, img_shape=(rbm_shape, rbm_shape), tile_shape=(10, 10), tile_spacing=(0,0))
								ax.matshow(matrix_new)
								plt.pause(0.00001)


						log.info("Learnrate:",round(LEARNRATE_PRETRAIN,4))
						log.info("error",round(error_i,4))
						log.end() #ending the epoch


					log.end() #ending training the rbm 

				

				# define the weights
				self.weights  =  []
				for i in range(len(self.RBMs)):
					self.weights.append(self.RBMs[i].w.eval())

				if DO_SAVE_PRETRAINED:
					for i in range(len(self.weights)):
						np.savetxt(workdir+"/pretrain_data/"+"Pretrained-"+" %i "%i+str(time_now)+".txt", self.weights[i])
					log.out("Saved Pretrained under "+str(time_now))
			else:
				if not DO_LOAD_FROM_FILE:
					### load the pretrained weights
					self.weights=[]
					log.out("Loading Pretrained from file")
					for i in range(self.n_layers-1):
						self.weights.append(np.loadtxt(workdir+"/pretrain_data/"+"Pretrained-"+" %i "%i+PATHSUFFIX_PRETRAINED+".txt").astype(np.float32))
				else:
					### if loading from file is active the pretrained weights would get 
					### reloaded anyway so directly load them here
					self.weights=[]
					log.out("Loading from file")
					for i in range(self.n_layers-1):
						self.weights.append(np.loadtxt(data_dir+"/"+PATHSUFFIX+"/"+"w%i.txt"%(i)).astype(np.float32))
			log.end()
			log.reset()

	def read_logfile(self):
		log_buffer=[]
		log_dict = {}
		with open("logfile.txt","r") as logfile:
			for line in logfile:
				for i in range(len(line)):
					if line[i]==",":
						save=i
				value=line[save+1:-1]
				try:
					value=float(value)
				except:
					pass
				log_buffer.append([(line[0:save]),value])
		for i in range(len(log_buffer)):
			key   = log_buffer[i][0]
			value = log_buffer[i][1]
			log_dict[key] = value
		return log_dict

	def load_from_file(self,path,override_params=0):
		""" loads weights and biases from folder and sets 
		variables like learnrate and temperature to the values
		that were used in the last epoch"""
		global learnrate, temp, freerun_steps
		os.chdir(path)
		log.out("Loading data from:","...",path[-20:])

		self.w_np     = []
		self.w_np_old = []
		for i in range(self.n_layers-1):
			self.w_np.append(np.loadtxt("w%i.txt"%(i)))
			self.w_np_old.append(self.w_np[i])  #save weights for later comparison

		self.bias_np = []
		for i in range(self.n_layers):
			self.bias_np.append(np.loadtxt("bias%i.txt"%(i)))
		if override_params:
			# try:
			log.out("Overriding Values from save")
			
			## read save dict and log file and save as dicts
			sd = read_csv("save_dict.csv")
			log_dict = self.read_logfile()


			l_string = sd["Learnrate"].values[[sd["Learnrate"].notnull()]]
			l_string[-1]=l_string[-1].replace("["," ")
			l_string[-1]=l_string[-1].replace("]"," ")
			l_ = np.fromstring(l_string[-1][1:-1],sep=" ")

			t_ = sd["Temperature"].values[[sd["Temperature"].notnull()]]
			n_ = sd["Freerun_Steps"].values[[sd["Freerun_Steps"].notnull()]]
			train_epoch_ = sd["Train_Epoch"].values[[sd["Train_Epoch"].notnull()]]

			freerun_steps = n_[-1]
			temp          = t_[-1]
			learnrate     = l_
			self.epochs   = train_epoch_[-1]

			try:
				self.update = log_dict["Update"]
			except:
				log.info("No key 'update' in logfile found.")


			log.info("Epoch = ",self.epochs)
			log.info("l = ",learnrate)
			log.info("T = ",round(temp,5))
			log.info("N = ",freerun_steps)
			# except:
			# 	log.error("Error overriding: Could not find save_dict.csv")
		os.chdir(workdir)

	def import_(self):
		""" setting up the graph and setting the weights and biases tf variables to the 
		saved numpy arrays """
		log.out("loading numpy vars into graph")
		for i in range(self.n_layers-1):
			sess.run(self.w[i].assign(self.w_np[i]))
		for i in range(self.n_layers):
			sess.run(self.bias[i].assign(self.bias_np[i]))

	def layer_input(self, layer_i):
		""" calculate input of layer layer_i
		layer_i :: for which layer
		returns :: input for the layer - which are the probabilites
		"""
		if layer_i == 0:
			w = self.w[layer_i];
			if USE_DROPOUT:
				w *= self.dropout_matrix[layer_i]
			_input_ = sigmoid(tf.matmul(self.layer[layer_i+1], w,transpose_b=True) + self.bias[layer_i], self.temp_tf)			

		elif layer_i == self.n_layers-1:
			w = self.w[layer_i-1];
			if USE_DROPOUT:
				w *= self.dropout_matrix[layer_i-1]
			_input_ = sigmoid(tf.matmul(self.layer[layer_i-1],w) + self.bias[layer_i], self.temp_tf)
		
		else:
			w0 = self.w[layer_i-1];
			w1 = self.w[layer_i];
			if USE_DROPOUT:
				w0 *= self.dropout_matrix[layer_i-1]
				w1 *= self.dropout_matrix[layer_i]
			_input_ = sigmoid(tf.matmul(self.layer[layer_i-1],w0) 
								+ tf.matmul(self.layer[layer_i+1],w1,transpose_b=True) 
								+ self.bias[layer_i], 
						       self.temp_tf
							  )
		return _input_

	def sample(self,x):
		""" takes sample from x where x is a probability vector.
		subtracts a random uniform from each prob and then applies the 
		sign function to just get +1,-1 , then a relu is applied to set 
		every elelemt with negative sign to 0
		"""
		# return tf.cast(x > tf.random_uniform(tf.shape(x)),tf.float32) 
		return	tf.nn.relu(
					tf.sign(
						x - tf.random_uniform(tf.shape(x))
					)
				)

	def glauber_step(self, clamp, temp, droprate, save_array, step):
		""" 
		Updates layer asynchronously (glauber dynamic), some layers can be set to be clamped 
		
		temp :: temperature 
		
		clamp :: string type. weather to clamp visible layer ("visible","v") or 
				label layer ("label","l") or visible and label ("visible+label","v+l") 
				for clamped train run or No clamp ("None") for the train free run
		
		save_array :: "None" or array of shape [layer,timestep,batchsize,n_units] to save every step in
		
		step :: current timestep

		returns :: list of resulting unit states as numpy arrays
		"""
		layer = [None]*self.n_layers
		if clamp == "visible+label" or clamp == "v+l":
			rnd_order = list(range(1,self.n_layers-1))
		elif clamp == "None":
			rnd_order = list(range(0,self.n_layers))
		elif clamp == "visible" or clamp == "v":
			rnd_order = list(range(1,self.n_layers))
		elif clamp == "label" or clamp == "l":
			rnd_order = list(range(0,self.n_layers-1))
		
		
		# shuffle the list to really make it random
		rnd.shuffle(rnd_order)

		# run the updates in that random order
		for layer in rnd_order:
			if save_array == "None":
				sess.run(self.update_l_s[layer], {self.temp_tf : temp, self.droprate_tf : droprate})
			else:
				save_array[layer][step] = sess.run(self.update_l_s[layer], {self.temp_tf : temp, self.droprate_tf : droprate})
		
		# delete empty layer elements
		# if clamp == "visible" or clamp == "v":
		# 	layer.pop(0)
		# elif clamp == "label" or clamp == "l":
		# 	layer.pop(-1)
		# elif clamp == "visible+label" or clamp == "v+l":
		# 	layer.pop(-1)
		# 	layer.pop(0)


		# return layer
				
	def get_learnrate(self,epoch,a,y_off):
		""" calculate the learnrate dependend on parameters
		epoch :: current epoch 
		a :: slope
		y_off :: y offset -> start learningrate
		"""
		L = np.zeros(len(y_off))
		for i in range(len(y_off)):
			L[i] = a / (float(a)/y_off[i]+epoch)
		return L

	def get_temp(self,update,a,y_off, t_min):
		""" calculate the Temperature dependend on parameters 
		update :: current update 
		a :: slope
		y_off :: y offset -> start learningrate
		"""
		T = y_off-float(a)*update
		if T < t_min:
			T = t_min;
		return T

	def get_N(self,epoch):
		# N = epoch*1
		# if N<4:
		# 	N = 4
		# if N>50:
		N = 10
		return int(N)

	def update_savedict(self,mode):
		if mode=="training":
			# append all data to save_dict
			self.save_dict["Train_Epoch"].append(self.epochs)
			self.save_dict["Temperature"].append(temp)
			self.save_dict["Learnrate"].append(learnrate)
			self.save_dict["Freerun_Steps"].append(freerun_steps)
			
			for i in range(self.n_layers-1):
				w_mean = sess.run(self.len_w[i])
				CD_mean = sess.run(self.CD_abs_mean[i])
				w_diff_mean = np.mean(self.weights_diff[i])
				self.save_dict["W_diff_%i"%i].append(w_diff_mean)	
				self.save_dict["W_mean_%i"%i].append(w_mean)				
				self.save_dict["CD_abs_mean_%i"%i].append(CD_mean)

			for i in range(self.n_layers):
				bias_diff_mean = np.mean(self.bias_diff[i])
				self.save_dict["Bias_diff_%i"%i].append(bias_diff_mean)
				self.save_dict["Layer_Diversity_%i"%i].append(self.layer_diversity_train[-1][i])


		if mode == "testing":
			if self.classification:
					self.save_dict["Class_Error"].append(self.class_error_test)
			self.save_dict["Recon_Error"].append(self.recon_error)
			self.save_dict["Test_Epoch"].append(self.epochs)


		if mode=="testing_with_train_data":
			if self.classification:
				self.save_dict["Class_Error_Train_Data"].append(self.class_error_test)

	def get_total_layer_input(self):
		l_input = [[[],[]] for _ in range(self.n_layers)]
		l_var = [[[],[]] for _ in range(self.n_layers)]
		for i in range(self.n_layers-1):
			a_mean, a_var = sess.run(tf.nn.moments(tf.abs(tf.matmul(self.layer[i],  self.w[i])), 0))
			b_mean, b_var = sess.run(tf.nn.moments(tf.abs(tf.matmul(self.layer[i+1],self.w[i],transpose_b=True)), 0))
			# means
			l_input[i+1][0].append(a_mean)
			l_input[i][1].append(b_mean)
			# var s
			l_var[i+1][0].append(a_var)
			l_var[i][1].append(b_var)	
		return l_input,l_var
	
	def get_units_input(self):
		### unit input histogram measure
		hist_input = [[[],[]] for _ in range(self.n_layers)]
		for i in range(self.n_layers-1):
			left_input = sess.run(tf.matmul(self.layer[i],  self.w[i]))
			right_input = sess.run(tf.matmul(self.layer[i+1],self.w[i],transpose_b=True))

			hist_input[i+1][0].append(left_input)
			hist_input[i][1].append(right_input)
		return hist_input

	def graph_init(self,graph_mode):
		""" sets the graph up and loads the pretrained weights in , these are given
		at class definition
		graph_mode  :: "training" if the graph is used in training - this will define more variables that are used in training - is slower
					:: "testing"  if the graph is used in testing - this will define less variables because they are not needed in testing
		"""
		log.out("Initializing graph")
		

		if graph_mode=="training":
			self.learnrate = tf.placeholder(tf.float32,[self.n_layers-1],name="Learnrate")

		#### temperature
		self.temp_tf = tf.placeholder(tf.float32, [], name="Temperature")
		self.droprate_tf = tf.placeholder(tf.float32, [], name="DropoutRate")


		### init all Parameters like weights , biases , layers and their updates
		## weights
		self.w               = [None]*(self.n_layers-1)
		self.pos_grad        = [None]*(self.n_layers-1)
		self.neg_grad        = [None]*(self.n_layers-1)
		self.update_pos_grad = [None]*(self.n_layers-1)
		self.update_neg_grad = [None]*(self.n_layers-1)
		self.update_w        = [None]*(self.n_layers-1)
		self.w_mean_         = [None]*(self.n_layers-1) # variable to store means
		self.len_w           = [None]*(self.n_layers-1) # calc of mean for each w
		self.do_norm_w       = [None]*(self.n_layers-1)
		self.dropout_matrix  = [None]*(self.n_layers-1) # store 0 and 1 randomly for each neuron connection
		self.CD              = [None]*(self.n_layers-1)
		self.CD_abs_mean     = [None]*(self.n_layers-1)


		# bias
		self.bias        = [None]*self.n_layers
		self.update_bias = [None]*self.n_layers
		self.mean_bias   = [None]*self.n_layers
		self.do_norm_b   = [None]*self.n_layers

		# layer
		self.layer             = [None]*self.n_layers # layer variable 
		self.layer_save        = [None]*self.n_layers # save variable (used for storing older layers)
		self.assign_save_layer = [None]*self.n_layers # save variable (used for storing older layers)
		# self.layer_particles   = [None]*self.n_layers # store the sum of multiple free runs 
		# self.update_particles  = [None]*self.n_layers # 
		# self.reset_particles   = [None]*self.n_layers # 
		self.layer_ph          = [None]*self.n_layers # placeholder 
		self.assign_l          = [None]*self.n_layers # assign op. (assigns placeholder)
		self.assign_l_rand     = [None]*self.n_layers # assign op. (assigns random)
		self.layer_prob        = [None]*self.n_layers # calc prob for layer n
		self.layer_samp        = [None]*self.n_layers # take a sample from the prob
		self.update_l_s        = [None]*self.n_layers # assign op. for calculated samples
		self.update_l_p        = [None]*self.n_layers # assign op. for calculated probs
		self.layer_activities  = [None]*self.n_layers # calc for layer activieties (mean over batch)
		self.layer_energy      = [None]*(self.n_layers-1)
		self.unit_diversity    = [None]*self.n_layers # measure how diverse each unit is in the batch 
		self.layer_diversity   = [None]*self.n_layers # measure how diverse each layer is in the batch 
		self.freerun_diff      = [None]*self.n_layers # calculates mean(abs(layer_save (clamped run) - layer)) 
														# if called after freerunnning it tells the mean difference between freeerun and clamped run

		### layer vars 
		for i in range(len(self.layer)):
			self.layer[i]      = tf.Variable(tf.random_uniform([self.batchsize,self.SHAPE[i]],minval=-1e-3,maxval=1e-3),name="Layer_%i"%i)
			self.layer_save[i] = tf.Variable(tf.random_uniform([self.batchsize,self.SHAPE[i]],minval=-1e-3,maxval=1e-3),name="Layer_save_%i"%i)
			# self.layer_particles[i] = tf.Variable(tf.zeros([self.batchsize,self.SHAPE[i]]),name="Layer_particle_%i"%i)
			self.layer_ph[i]   = tf.placeholder(tf.float32,[self.batchsize,self.SHAPE[i]],name="layer_%i_PH"%i)

		### weight calculations and assignments
		for i in range(len(self.w)):
			self.w[i] = tf.Variable(self.weights[i],name="Weights%i"%i)
		for i in range(len(self.w)):
			if graph_mode=="training":
				self.pos_grad[i]        = tf.Variable(tf.zeros([self.SHAPE[i],self.SHAPE[i+1]]))
				self.neg_grad[i]        = tf.Variable(tf.zeros([self.SHAPE[i],self.SHAPE[i+1]]))
				self.update_pos_grad[i] = self.pos_grad[i].assign(tf.matmul(self.layer[i], self.layer[i+1], transpose_a=True))
				self.update_neg_grad[i] = self.neg_grad[i].assign(tf.matmul(self.layer[i], self.layer[i+1], transpose_a=True))
				self.CD[i]              = self.pos_grad[i] - self.neg_grad[i]
				self.CD_abs_mean[i]     = tf.reduce_mean(tf.abs(self.CD[i]))
				self.update_w[i]        = self.w[i].assign_add(self.learnrate[i]*temp/self.batchsize*self.CD[i]-lambda_learn)
				self.w_mean_[i]         = tf.Variable(tf.zeros([N_EPOCHS_TRAIN]))
				self.len_w[i]          = tf.sqrt(tf.reduce_sum(tf.square(self.w[i])))
				
				## old norm approach
				self.do_norm_w[i] = self.w[i].assign(self.w[i]/self.len_w[i])

				# droprate is constant- used for training
				self.dropout_matrix[i]  = tf.round(tf.clip_by_value(tf.random_uniform(tf.shape(self.w[i]))*DROPOUT_RATE,0,1))
			else:
				# droprate is variable and can be decreased over time
				self.dropout_matrix[i] = tf.round(tf.clip_by_value(tf.random_uniform(tf.shape(self.w[i]))*self.droprate_tf,0,1))

		### bias calculations and assignments
		for i in range(len(self.bias)):
			self.bias[i] = tf.Variable(tf.zeros([self.SHAPE[i]]),name="Bias%i"%i)
			if graph_mode == "training":
				self.update_bias[i] = self.bias[i].assign_add(self.learnrate[0]*temp*tf.reduce_mean(tf.subtract(self.layer_save[i],self.layer[i]),0)-lambda_learn)
				self.mean_bias[i]   = tf.sqrt(tf.reduce_sum(tf.square(self.bias[i])))
				bias_len            = abs_norm(self.bias[i],0)
				
				## old norm approach
				self.do_norm_b[i]   = self.bias[i].assign(self.bias[i]/self.mean_bias[i])

				### new norm approach
				# self.do_norm_b[i]   = self.bias[i].assign(self.bias[i]/tf.where(bias_len<0.1, tf.ones_like(bias_len), bias_len))

		### layer calculations and assignments
		for i in range(len(self.layer)):
			self.assign_save_layer[i] = self.layer_save[i].assign(self.layer[i])
			self.assign_l[i]          = self.layer[i].assign(self.layer_ph[i])
			self.assign_l_rand[i]     = self.layer[i].assign(tf.random_uniform([self.batchsize,self.SHAPE[i]]))
			self.layer_prob[i]        = self.layer_input(i)
			self.layer_samp[i]        = self.sample(self.layer_prob[i])
			self.update_l_p[i]        = self.layer[i].assign(self.layer_prob[i])
			self.layer_activities[i]  = tf.reduce_sum(self.layer[i])/(self.batchsize*self.SHAPE[i])*100
			self.unit_diversity[i]    = tf.sqrt(tf.reduce_mean(tf.square(self.layer[i] - tf.reduce_mean(self.layer[i], axis=0)),axis=0))
			self.layer_diversity[i]   = tf.reduce_mean(self.unit_diversity[i])
			self.freerun_diff[i]      = tf.reduce_mean(tf.abs(self.layer_save[i]-self.layer[i]))

		for i in range(len(self.layer)-1):
			if i <len(self.layer)-2:
				self.layer_energy[i] = tf.einsum("ij,ij->i",self.layer[i+1], tf.matmul(self.layer[i],self.w[i]))+tf.reduce_sum(self.layer[i]*self.bias[i],1)
			else:
				self.layer_energy[i] = tf.einsum("ij,ij->i",self.layer[i+1], tf.matmul(self.layer[i],self.w[i]))+tf.reduce_sum(self.layer[i]*self.bias[i],1)+tf.reduce_sum(self.layer[i+1]*self.bias[i+1],1)
			self.update_l_s[i]   = self.layer[i].assign(self.layer_samp[i])
		self.update_l_s[-1] = self.layer[-1].assign(self.layer_prob[-1])#self.layer[-1].assign(self.layer_prob[-1])

		# modification array size 10 that gehts multiplied to the label vector for context
		# self.modification_tf = tf.Variable(tf.ones([self.batchsize,self.SHAPE[-1]]),name="Modification")


		### Error and stuff
		self.error       = tf.reduce_mean(tf.square(self.layer_ph[0]-self.layer[0]))
		self.class_error = tf.reduce_mean(tf.square(self.layer_ph[-1]-self.layer[-1]))

		self.free_energy = -tf.reduce_sum(tf.log(1+tf.exp(tf.matmul(self.layer[0],self.w[0])+self.bias[1])))

		self.energy = -tf.add_n([self.layer_energy[i] for i in range(len(self.layer_energy))])

		

		sess.run(tf.global_variables_initializer())
		self.init_state=1

	def test_noise_stability(self,input_data,input_label,steps):
		self.batchsize=len(input_data)
		if DO_LOAD_FROM_FILE:
			self.load_from_file(workdir+"/data/"+PATHSUFFIX)
		self.graph_init("testing")
		self.import_()

		n       = 20
		h2_     = []
		r       = rnd.random([self.batchsize,784])
		v_noise = np.copy(input_data)
		# make the input more noisy
		v_noise += (abs(r-0.5)*0.5)
		v_noise = sample_np(v_noise)

		sess.run(self.assign_l[0] , {self.layer_ph[0] : v_noise})
		sess.run(self.update_l_p[1], {self.temp_tf : temp})
		sess.run(self.update_l_p[2], {self.temp_tf : temp})

		for i in range(steps):
			
			layer = sess.run(self.update_l_s, {self.temp_tf : temp})
			

			if self.classification:
				h2_.append(layer[-1])


		v_noise_recon = sess.run(self.update_l_p[0], {self.temp_tf : temp})
		return np.array(h2_),v_noise_recon,v_noise

	def train(self,train_data,train_label,num_batches,cont):
		global learnrate, temp, freerun_steps
		""" training the DBM with given h2 as labels and v as input images
		train_data  :: images
		train_label :: corresponding label
		num_batches :: how many batches
		"""
		######## init all vars for training
		self.batchsize = int(len(train_data)/num_batches)
		self.num_of_updates = N_EPOCHS_TRAIN*num_batches


		# number of clamped sample steps
		if self.n_layers <=3 and self.classification==1:
			M = 2
		else:
			M = 50




		### free energy
		# self.F=[]
		# self.F_test=[]
		
		if DO_LOAD_FROM_FILE and not cont:
			# load data from the file
			self.load_from_file(workdir+"/data/"+PATHSUFFIX,override_params=1)
			self.graph_init("training")
			self.import_()

		# if no files loaded then init the graph with pretrained vars
		if self.init_state==0:
			self.graph_init("training")

		if cont and self.tested:
			self.graph_init("training")
			self.import_()
			self.tested = 0


		if self.liveplot:
			log.info("Liveplot is on!")
			fig,ax = plt.subplots(1,1,figsize=(15,10))
			data   = ax.matshow(tile(self.w[0].eval()), vmin=-0.01, vmax=0.01)
			plt.colorbar(data)

		## save the old weights and biases
		w_old = []
		b_old = []
		for i in range(self.n_layers):
			if i < self.n_layers-1:
				w_old.append(np.copy(self.w[i].eval()))
			b_old.append(np.copy(self.bias[i].eval()))

		# starting the training
		log.info("Batchsize:",self.batchsize,"N_Updates",self.num_of_updates)
		
		log.start("Deep BM Epoch:",self.epochs+1,"/",N_EPOCHS_TRAIN)

		# shuffle test data and labels so that batches are not equal every epoch 
		log.out("Shuffling TrainData")
		self.seed   = rnd.randint(len(train_data),size=(int(len(train_data)/10),2))
		train_data  = shuffle(train_data, self.seed)
		if self.classification:
			train_label = shuffle(train_label, self.seed)

		log.out("Running Epoch")
		# log.info("++ Using Weight Decay! Not updating bias! ++")
		# self.persistent_layers = sess.run(self.update_l_s,{self.temp_tf : temp})
		
		for start, end in zip( range(0, len(train_data), self.batchsize), range(self.batchsize, len(train_data), self.batchsize)):
			# define a batch
			batch = train_data[start:end]
			if self.classification:
				batch_label = train_label[start:end]




			#### Clamped Run 

			# assign v and h2 to the batch data
			sess.run(self.assign_l[0], { self.layer_ph[0]  : batch })
			if self.classification:
				sess.run(self.assign_l[-1], {self.layer_ph[-1] : batch_label})

			# calc hidden layer samples (not the visible & label layer)
			for hidden in range(M):
				if self.classification:
					sess.run(self.update_l_s[1:-1],{self.temp_tf : temp})#self.glauber_step(clamp="v+l",temp=temp) #sess.run(self.update_l_s[1:-1],{self.temp_tf : temp})
				else:
					sess.run(self.update_l_s[1:],{self.temp_tf : temp})#self.glauber_step(clamp="v",temp=temp) #sess.run(self.update_l_s[1:],{self.temp_tf : temp})

			# last run calc only the probs to reduce noise
			sess.run(self.update_l_p[1:-1],{self.temp_tf : temp})
			# save all layer for bias update
			sess.run(self.assign_save_layer)
			# update the positive gradients
			sess.run(self.update_pos_grad)


			#### Free Running 

					# update all layers N times (Gibbs sampling)
					# sess.run(self.reset_particles)
					
					# for p in range(self.n_particles):
					# 	sess.run(self.assign_l[0], {self.layer_ph[0]   : batch})
					# 	for i in range(self.n_layers-2):
					# 		sess.run(self.assign_l[i+1], {self.layer_ph[i+1]   : ls_[i]})
					# 	sess.run(self.assign_l[-1],{self.layer_ph[-1]  : batch_label})
						
			for n in range(freerun_steps):
				sess.run(self.update_l_s,{self.temp_tf : temp})#self.glauber_step(clamp = "None",temp=temp) #sess.run(self.update_l_s,{self.temp_tf : temp})

	
			
			# calc probs for noise cancel				
			sess.run(self.update_l_p,{self.temp_tf : temp})
			# sess.run(self.update_particles)
		

			# calc he negatie gradients
			sess.run(self.update_neg_grad)

			
			#### run all parameter updates 
			sess.run([self.update_w, self.update_bias], {self.learnrate : learnrate})
			

			#### norm the weights
			if DO_NORM_W:
				## simple method
				for i in range(self.n_layers):
					# weights 
					if i<(self.n_layers-1) and sess.run(self.len_w[i]) > 1:
						sess.run(self.do_norm_w[i])
					# bias 
					if sess.run(self.mean_bias[i]) > 1:
						sess.run(self.do_norm_b[i])

				## other method (does not work)
				# sess.run(self.do_norm_w)
				# sess.run(self.do_norm_b)


			### calc errors and other things
			if self.update%10==0:
				self.recon_error_train[self.update/10] = (sess.run(self.error,{self.layer_ph[0] : batch}))
				if self.classification:
					self.class_error_train[self.update/10] = (sess.run(self.class_error,{self.layer_ph[-1] : batch_label}))
				self.layer_diversity_train[self.update/10] = (sess.run(self.layer_diversity))
				self.layer_act_train[self.update/10] = (sess.run(self.layer_activities))

				self.l_mean += sess.run(self.layer_activities)
				
				# check if freerunning escaped fixpoint
				self.freerun_diff_train[self.update/10] = sess.run(self.freerun_diff)
			
			
			## update parameters
			self.update += 1

			temp = self.get_temp(self.update, TEMP_SLOPE, TEMP_START, TEMP_MIN)

			### liveplot
			if self.liveplot and plt.fignum_exists(fig.number) and start%40==0:
				if start%4000==0:
					ax.cla()
					data = ax.matshow(tile(self.w[0].eval()),vmin=tile(self.w[0].eval()).min()*1.2,vmax=tile(self.w[0].eval()).max()*1.2)

				matrix_new = tile(self.w[0].eval())
				data.set_data(matrix_new)
				plt.pause(0.00001)

		
		log.end() #ending the epoch
	
		
		
		# export the tensorflow vars into numpy arrays
		self.export()
			
		
		# calculate diff between all vars 
		log.out("Calculation weights diff")
		self.weights_diff = []
		self.bias_diff    = []
		for i in range(self.n_layers):
			if i < self.n_layers-1:
				self.weights_diff.append(np.abs(self.w_np[i]-w_old[i]))
			self.bias_diff.append(np.abs(self.bias_np[i]-b_old[i]))


		### write vars into savedict
		self.update_savedict("training")
		self.l_mean[:] = 0

		# increase epoch counter
		self.epochs += 1 
		
		# change learnrate
		log.info("Learnrate: ",np.round(learnrate,6))
		learnrate = self.get_learnrate(self.epochs, LEARNRATE_SLOPE, LEARNRATE_START)
		
		# print temp
		log.info("Temp: ",np.round(temp,5))
			# learnrate change is inside batch loop

		# change freerun_steps
		log.info("freerun_steps: ",freerun_steps)
		freerun_steps = self.get_N(self.epochs)

		# average layer activities over epochs 
		self.l_mean *= 1.0/num_batches



		log.reset()

	def test(self,my_test_data,my_test_label,N,create_conf_mat, temp_start, temp_end, using_train_data = False):
		""" 
		testing runs without giving h2 , only v is given and h2 has to be infered 
		by the DBM 
		array my_test_data :: images to test, get assigned to v layer
		int N :: Number of updates from hidden layers 
		
		"""


		### init the vars and reset the weights and biases 		
		self.batchsize       = len(my_test_data)
		self.learnrate       = LEARNRATE_START
		
		self.layer_save_test = [[None] for i in range(self.n_layers)]   # save layers while N hidden updates
		for layer in range(len(self.layer_save_test)):
			self.layer_save_test[layer] = np.zeros([N, self.batchsize, self.SHAPE[layer]])
		self.layer_act_test  = np.zeros([N,self.n_layers])  # layer activities while N hidden updates

		#temp decrease
		temp                 = temp_start
		temp_delta           = (temp_end-temp_start)/float(N)

		droprate = 100 # 100 => basically no dropout

		### init the graph 
		if DO_LOAD_FROM_FILE and not DO_TRAINING:
			self.load_from_file(workdir+"/data/"+PATHSUFFIX,override_params=1)
		self.graph_init("testing") # "testing" because this graph creates the testing variables where only v is given, not h2
		self.import_()


		#### start test run
		log.start("Testing DBM with %i images"%self.batchsize)
		if using_train_data:
			log.info("Using train data")
		else:
			log.info("Using test data")

		# give input to v layer
		sess.run(self.assign_l[0], {self.layer_ph[0] : my_test_data, self.temp_tf : temp})

		# update hidden and label N times
		log.start("Sampling hidden %i times "%N)
		log.info("temp: %f -> %f"%(np.round(temp_start,5),np.round(temp_end,5)))		
		# make N clamped updates
		for n in range(N):
			self.layer_act_test[n,:] = sess.run(self.layer_activities, {self.temp_tf : temp})
			self.glauber_step("visible", temp, droprate, self.layer_save_test, n) # sess.run(self.update_l_s[1:], {self.temp_tf : temp})
			# increment temp
			temp+=temp_delta

		# calc layer variance across batch
		self.layer_diversity_test = sess.run(self.layer_diversity)

		

		log.end()

		## get firerates of every unit 
		self.firerate_test = sess.run(self.update_l_p, {self.temp_tf : temp, self.droprate_tf : droprate})
		# were firerates are around 0.1
		research_layer = 3 
		self.neuron_index_test = np.where((np.mean(DBM.firerate_test[research_layer],0)>0.08) & (np.mean(DBM.firerate_test[research_layer],0)<0.12))[0]
		# generate hisogramms for all these neurons 
		self.hists_test = calc_neuron_hist(self.neuron_index_test,DBM.firerate_test[research_layer],test_label,0.5)

		### layer input measure from each adjacent layer
		self.l_input_test, self.l_var_test = self.get_total_layer_input()
		### unit input histogram measure
		self.hist_input_test =  self.get_units_input()

		
		#### reconstruction of v 
		# update v M times
		self.last_layer_save = self.layer_save_test[-1][-1]

		for i in range(N):
			self.layer_save_test[0][i] = sess.run(self.update_l_s[0],{self.temp_tf : temp, self.droprate_tf: droprate})




		#### calculate errors and activations
		self.recon_error  = self.error.eval({self.layer_ph[0] : my_test_data})
		

		#### count how many images got classified wrong 
		log.out("Taking only the maximum")
		n_wrongs             = 0
		# label_copy         = np.copy(self.last_layer_save)
		wrong_classified_ind = []
		wrong_maxis          = []
		right_maxis          = []
		

		if self.classification:
			## error of classifivation labels
			self.class_error_test = np.mean(np.abs(self.last_layer_save-my_test_label[:,:10]))		
			
			for i in range(len(self.last_layer_save)):
				digit   = np.where(my_test_label[i]==1)[0][0]
				maxi    = self.last_layer_save[i].max()
				max_pos = np.where(self.last_layer_save[i] == maxi)[0][0]
				if max_pos != digit:
					wrong_classified_ind.append(i)
					wrong_maxis.append(maxi)#
				elif max_pos == digit:
					right_maxis.append(maxi)
			n_wrongs = len(wrong_maxis)

			if create_conf_mat:
				log.out("Making Confusion Matrix")
				
				self.conf_data = np.zeros([10,1,10]).tolist()

				for i in range(self.batchsize):
					digit = np.where( test_label[i] == 1 )[0][0]
					
					self.conf_data[digit].append( self.last_layer_save[i].tolist() )

				# confusion matrix
				w = np.zeros([10,10])
				for digit in range(10):
					w[digit]  = np.round(np.mean(np.array(DBM.conf_data[digit]),axis=0),3)
				seaborn.heatmap(w*100,annot=True)
				plt.ylabel("Desired Label in %")
				plt.xlabel("Predicted Label in %")
								
		self.class_error_test = float(n_wrongs)/self.batchsize


		
		# append test results to save_dict
		if not using_train_data:
			self.update_savedict("testing")
		elif using_train_data:
			self.update_savedict("testing_with_train_data")


		self.tested = 1 # this tells the train function that the batchsize has changed
		
		log.end()
		log.info("------------- Test Log -------------")
		log.info("Reconstr. error normal: ",np.round(self.recon_error,5))
		# if self.n_layers==2: log.info("Reconstr. error reverse: ",np.round(self.recon_error_reverse,5)) 
		if self.classification:
			log.info("Class error: ",np.round(self.class_error_test, 5))
			log.info("Wrong Digits: ",n_wrongs," with average: ",round(np.mean(wrong_maxis),3))
			log.info("Correct Digits: ",len(right_maxis)," with average: ",round(np.mean(right_maxis),3))
		log.reset()
		return wrong_classified_ind

	def gibbs_sampling(self, v_input, gibbs_steps, TEMP_START, temp_end, droprate_start, droprate_end, subspace, mode, liveplot=1):
		""" Repeatedly samples v and label , where label can be modified by the user with the multiplication
		by the modification array - clamping the labels to certain numbers.
		v_input :: starting with an image as input can also be a batch of images
		
		temp_end, TEMP_START :: temperature will decrease or increase to temp_end and start at TEMP_START 
		
		mode 	:: "generate"  clamps label
				:: "freerunnning" clamps nothing
				:: "context" clamps v and only calculates h1 based on previous h2
		
		subspace :: {"all", array} if "all" do nothing, if array: set the weights to 0 for all indices not marked by subspace
		 		used with "context" mode for clamping certain labels to 0
		"""

		self.layer_save = []
		for i in range(self.n_layers):
			self.layer_save.append(np.zeros([gibbs_steps,self.batchsize,self.SHAPE[i]]))
	
		layer_gs = [None]*self.n_layers
		for l in range(len(layer_gs)):
			layer_gs[l] = np.zeros([gibbs_steps,self.batchsize,self.SHAPE[l]])
		
		temp_          = np.zeros([gibbs_steps])

		self.energy_   = []
		self.mean_h1   = []

		temp           = TEMP_START
		temp_delta     = (temp_end-TEMP_START)/gibbs_steps

		droprate   = droprate_start
		drop_delta = (droprate_end-droprate_start)/gibbs_steps 

		self.num_of_updates = 1000 #just needs to be defined because it will make a train graph with tf.arrays where this number is needed


		if liveplot:
			log.info("Liveplotting gibbs sampling")
			fig,ax=plt.subplots(1,self.n_layers+1,figsize=(15,6))
			# plt.tight_layout()

		log.start("Gibbs Sampling")
		log.info("Mode: %s | Steps: %i"%(mode,gibbs_steps))
		log.info("Temp_range:",round(TEMP_START,5),"->",round(temp_end,5))
		log.info("Dropout_range:",round(droprate_start,5),"->",round(droprate_end,5))

		if mode=="context":
			sess.run(self.assign_l[0],{self.layer_ph[0] : v_input})
			for i in range(1,self.n_layers):
				sess.run( self.assign_l[i], {self.layer_ph[i] : 0.01*rnd.random([self.batchsize, self.SHAPE[i]])} )
			

			input_label = test_label[index_for_number_gibbs[:]]


			# set the weights to 0 if context is enebaled and subspace is not "all"
			if subspace == "all":
				pass
				self.activity_nc  = np.zeros([self.n_layers-1,gibbs_steps]);
				self.layer_diff_gibbs_nc = np.zeros([self.n_layers-1,gibbs_steps])
				self.class_error_gibbs_nc = np.zeros([gibbs_steps]);
			else:
				self.activity_c   = np.zeros([self.n_layers-1,gibbs_steps]);
				self.layer_diff_gibbs_c = np.zeros([self.n_layers-1,gibbs_steps])
				self.class_error_gibbs_c = np.zeros([gibbs_steps]);
				# get all numbers that are not in subspace
				subspace_anti = []
				for i in range(10):
					if i not in subspace:
						subspace_anti.append(i)
				

				log.out("Setting Weights to 0")
				# get the weights as numpy arrays
				w_ = self.w[-1].eval()
				b_ = self.bias[-1].eval()
				# set values to 0
				w_[:,subspace_anti] = 0
				b_[subspace_anti] = -1000
				# assign to tf variables
				sess.run(self.w[-1].assign(w_))
				sess.run(self.bias[-1].assign(b_))



			### gibbs steps
			for step in range(gibbs_steps):

				# update all self.layer except first one (set step = 0 because time series is not saved)
				self.glauber_step("visible", temp, droprate, layer_gs, step) #sess.run(self.update_l_s[1:], {self.temp_tf : temp})
				
				
				if subspace == "all":
					## without context
					#	save layer activites 
					self.activity_nc[:,step]  = sess.run(self.layer_activities[1:], {self.temp_tf : temp})
					self.class_error_gibbs_nc[step] = np.mean(np.abs(layer_gs[-1][step]-input_label))
					# save layer difference to previous layers
					if step != 0:
						for i in range(1,self.n_layers):
							self.layer_diff_gibbs_nc[i-1, step] = np.mean(np.abs(layer_gs[i][step-1] - layer_gs[i][step]))
				else:
					## wih context
					# save layer activites 
					self.activity_c[:,step]   = sess.run(self.layer_activities[1:], {self.temp_tf : temp})
					self.class_error_gibbs_c[step] = np.mean(np.abs(layer_gs[-1][step]-input_label))
					# save layer difference to previous layers
					if step!=0:
						for i in range(1,self.n_layers):
							self.layer_diff_gibbs_c[i-1, step] = np.mean(np.abs(layer_gs[i][step-1] - layer_gs[i][step]))



				# assign new temp and dropout rate
				temp += temp_delta 
				droprate += drop_delta

			
			
			
			## gather input data # calc layer variance across batch
			if subspace=="all":
				## calc layer probs and set these to the layer vars to smooth later calcs
				self.firerates_nc              = sess.run(self.update_l_p[1:],{self.temp_tf : temp, self.droprate_tf : droprate})
				self.l_input_nc, self.l_var_nc = self.get_total_layer_input()
				self.hist_input_nc             = self.get_units_input()
				self.unit_diversity_nc         = sess.run(self.unit_diversity)
				self.layer_diversity_nc        = sess.run(self.layer_diversity)
			else:
				self.firerates_c             = sess.run(self.update_l_p[1:],{self.temp_tf : temp, self.droprate_tf : droprate})
				self.l_input_c, self.l_var_c = self.get_total_layer_input()
				self.hist_input_c            = self.get_units_input()
				self.unit_diversity_c        = sess.run(self.unit_diversity)
				self.layer_diversity_c       = sess.run(self.layer_diversity)


		if mode=="generate":
			sess.run(self.assign_l_rand)
			sess.run(self.layer[-1].assign(v_input))


			## init save arrays for every layer and every gibbs step
			self.layer_save_generate = [[None] for i in range(self.n_layers)]
			for layer in range(len(self.layer_save_generate)):
				self.layer_save_generate[layer] = np.zeros( [gibbs_steps, self.batchsize, self.SHAPE[layer]] )
			self.energy_generate = np.zeros([gibbs_steps,self.batchsize])

			for step in range(gibbs_steps):
				# update all layer except the last one (labels) 
				self.glauber_step("label",temp, droprate, self.layer_save_generate, step) #sess.run(self.update_l_s[:-1], {self.temp_tf : temp})
				self.energy_generate[step] = sess.run(self.energy)

				# save layers 
				if liveplot:
					for layer_i in range(len(layer)):
						self.layer_save[layer_i][step] = layer[layer_i]
					self.layer_save[-1][step] = v_input
					# calc the energy
					self.energy_.append(sess.run(self.energy))
					# save values to array
					temp_[step] = temp
				

				# assign new temp
				temp += temp_delta 
				droprate+=drop_delta
		
		if mode=="freerunning":
			sess.run(self.assign_l_rand)
			rng  =  rnd.randint(100)
			# sess.run(self.assign_l[0], {self.layer_ph[0] : test_data[rng:rng+1]})
			# for i in range(10):
			# 	sess.run(self.update_l_s[1:],{self.temp_tf : temp})
			
			## init save arrays for every layer and every gibbs step
			self.layer_save_generate = [[None] for i in range(gibbs_steps)]
			# for i in range(len(self.layer_save_generate)):
			# 	self.layer_save_generate[i] = np.zeros([gibbs_steps,DBM.SHAPE[i]])
			self.energy_generate = np.zeros([gibbs_steps,self.batchsize])
			

			for step in range(gibbs_steps):
				
				# update all layer 
				
				self.glauber_step("None", temp, droprate, self.layer_save_generate, step) #sess.run(self.update_l_s, {self.temp_tf : temp}) 
				self.energy_generate[step] = sess.run(self.energy)
				
				
				if liveplot:
					# ass_save_layer
					for layer_i in range(len(layer)):
						self.layer_save[layer_i][step] = layer[layer_i]
					# calc the energy
					self.energy_.append(sess.run(self.energy))
					# save values to array
					temp_[step] = temp
			
				# assign new temp
				temp += temp_delta 

	
		if liveplot and plt.fignum_exists(fig.number) and self.batchsize==1:
			data = [None]*(self.n_layers+1)
			ax[0].set_title("Visible Layer")
			for layer_i in range(len(self.layer_save)):
				s = int(sqrt(self.SHAPE[layer_i]))
				if s!=3:
					data[layer_i]  = ax[layer_i].matshow(self.layer_save[layer_i][0].reshape(s,s),vmin=0,vmax=1)
					ax[layer_i].set_xticks([])
					ax[layer_i].set_yticks([])

				# ax[layer_i].set_yticks([])
				# ax[layer_i].set_xticks([])
				# ax[layer_i].grid(False)
			
			if self.classification:
				data[-2], = ax[-2].plot([],[])
				ax[-2].set_ylim(0,1)
				ax[-2].set_xlim(0,10)
				ax[-2].set_title("Classification")

			data[-1], = ax[-1].plot([],[])
			ax[-1].set_xlim(0,len(self.energy_))
			ax[-1].set_ylim(np.min(self.energy_),0)
			ax[-1].set_title("Energy")

			
			for step in range(1,gibbs_steps-1,2):
				if plt.fignum_exists(fig.number):
					ax[1].set_title("Temp.: %s, Steps: %s"%(str(round(temp_[step],3)),str(step)))

					for layer_i in range(len(self.layer_save)):
						s = int(sqrt(self.SHAPE[layer_i]))
						if s!=3:
							data[layer_i].set_data(self.layer_save[layer_i][step].reshape(s,s))
					if self.classification:
						data[-2].set_data(range(10),self.layer_save[-1][step])
					data[-1].set_data(range(step),self.energy_[:step])
					
					plt.pause(1/50.)
		
			plt.close(fig)

		log.end()
		if mode=="freerunning" or mode=="generate":
			# return the last images that got generated 
			return np.mean(self.layer_save_generate[0][-40:],0)
			# v_layer = sess.run(self.update_l_p[0], {self.temp_tf : temp})
			# return v_layer

		else:
			# return the mean of the last 30 gibbs samples for all images
			return np.mean(layer_gs[-1][-30:,:],axis=0)

	def export(self):
		# convert weights and biases to numpy arrays
		self.w_np=[]
		for i in range(self.n_layers-1):
			self.w_np.append(self.w[i].eval())
		self.bias_np = []
		for i in range(self.n_layers):	
			self.bias_np.append(self.bias[i].eval())

		# convert tf.arrays to numpy arrays 
		# if training:
		# 	self.h1_activity_np = self.h1_activity_.eval()
		# 	self.h2_activity_np = self.h2_activity_.eval()
		# 	self.train_error_np = self.train_error_.eval()
		# 	self.train_class_error_np = self.train_class_error_.eval()
		# 	self.w_mean_np = []
		# 	for i in range(self.n_layers-1):
		# 		self.w_mean_np.append(self.w_mean_[i].eval())
		
		self.exported = 1
		log.info("Saved Weights and Biases as NumPy Arrays.")

	def backup_params(self):
		if DO_TRAINING:
			new_path = saveto_path+"/Backups/Backup_%i-Error_%f"%(self.epochs,np.round(self.class_error_test, 3))
			if not os.path.isdir(new_path):
				os.makedirs(new_path)
			os.chdir(new_path)
			
			if self.exported!=1:
				self.export()

			# save weights 
			for i in range(self.n_layers-1):
				np.savetxt("w%i.txt"%i, self.w_np[i])
			
			##  save bias
			for i in range(self.n_layers):
				np.savetxt("bias%i.txt"%i, self.bias_np[i])
			
			## save save_dict
			try:
				save_df = DataFrame(dict([ (k,Series(v)) for k,v in self.save_dict.iteritems() ]))
			except:
				log.out("using dataframe items conversion for python 3.x")
				save_df = DataFrame(dict([ (k,Series(v)) for k,v in self.save_dict.items() ]))
			save_df.to_csv("save_dict.csv")
		os.chdir(workdir)

	def write_to_file(self):
		new_path = saveto_path
		if not os.path.isdir(saveto_path):
			os.makedirs(new_path)
		os.chdir(new_path)

		if DO_TRAINING:
			if self.exported!=1:
				self.export()

			# save weights 
			for i in range(self.n_layers-1):
				np.savetxt("w%i.txt"%i, self.w_np[i])
			
			##  save bias
			for i in range(self.n_layers):
				np.savetxt("bias%i.txt"%i, self.bias_np[i])
			
			## save save_dict
			try:
				save_df = DataFrame(dict([ (k,Series(v)) for k,v in self.save_dict.iteritems() ]))
			except:
				log.out("using dataframe items conversion for python 3.x")
				save_df = DataFrame(dict([ (k,Series(v)) for k,v in self.save_dict.items() ]))
			save_df.to_csv("save_dict.csv")
		

		## save log
		self.log_list.append(["train_time",self.train_time])
		self.log_list.append(["Epochs",self.epochs])

		## logfile
		with open("logfile.txt","w") as log_file:
				for i in range(len(self.log_list)):
					log_file.write(self.log_list[i][0]+","+str(self.log_list[i][1])+"\n")
				log_file.write("Update"+","+str(self.update)+"\n")


		log.info("Saved data and log to:",new_path)
		os.chdir(workdir)


###########################################################################################################
#### User Settings ###

N_BATCHES_PRETRAIN = 500 				# how many batches per epoch for pretraining
N_BATCHES_TRAIN    = 500 				# how many batches per epoch for complete DBM training
N_EPOCHS_PRETRAIN  = [0,0,0,0,0,0] 	# pretrain epochs for each RBM
N_EPOCHS_TRAIN     = 2  				# how often to iter through the test images
TEST_EVERY_EPOCH   = 10 			# how many epochs to train before testing on the test data

### learnrates 
LEARNRATE_PRETRAIN = 0.01				# learnrate for pretraining
LEARNRATE_START    = [	0.01,
						0.01,
						0.01,
						0.01,
						]				# starting learnrates for each weight. Biases always use the [0] entry
LEARNRATE_SLOPE    = 1.					# bigger number -> smaller slope


lambda_learn = 0.00000 				# test param for sparsness, gets added to weight update


### temperature
TEMP_START    = 0.01				# starting temp
TEMP_SLOPE    = 0 #10e-7			# linear decrease slope higher number -> fast cooling
TEMP_MIN      = 0.01

### state vars 
DO_PRETRAINING = 1		# if no pretrain then files are automatically loaded
DO_TRAINING    = 1		# if to train the whole DBM
DO_TESTING     = 1		# if testing the DBM with test data
DO_SHOW_PLOTS  = 1		# if plots will show on display - either way they get saved into saveto_path

DO_CONTEXT    = 0		# if to test the context
DO_GEN_IMAGES = 0		# if to generate images (mode can be choosen at function call)
DO_NOISE_STAB = 0		# if to make a noise stability test

USE_DROPOUT  = 0		# if to use synnaptic failure while training
DROPOUT_RATE = 2		# multiplication of random uniform synaptic failure matrix (higher number -> less failure)

DO_NORM_W    = 1		# if to norm the weights and biases to 1 while training

### saving and loading
DO_SAVE_TO_FILE       = 0 	# if to save plots and data to file
DO_SAVE_PRETRAINED    = 0 	# if to save the pretrained weights seperately (for later use)
DO_LOAD_FROM_FILE     = 0	# if to load weights and biases from datadir + pathsuffix
PATHSUFFIX            = "Mon_Jun__4_15-55-25_2018_[784, 225, 225, 225, 10] - ['original'] 15%"
						#"Mon_Jun__4_15-55-25_2018_[784, 225, 225, 225, 10] - ['original'] 15%"
							#"Thu_Jun__7_16-21-28_2018_[784, 225, 225, 225, 10] - ['15cont4']"
PATHSUFFIX_PRETRAINED = "Thu_Jun__7_13-49-25_2018"


DBM_SHAPE = [	int(sqrt(len(train_data[0])))*int(sqrt(len(train_data[0]))),
				8*8,
				8*8,
				8*8,
				10]
###########################################################################################################




saveto_path=data_dir+"/"+time_now+"_"+str(DBM_SHAPE)

### modify the parameters with additional_args
if len(additional_args) > 0:
	# first_param = int(additional_args[0])
	# T = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
	# # LEARNRATE_START = [ l[first_param], l[first_param]]
	# TEMP_START = T[first_param]
	saveto_path  += " - " + str(additional_args)


### globals (to be set as DBM self values)
freerun_steps = 2 					# global number of freerun steps for training
learnrate     = LEARNRATE_START		# global learnrate
temp          = TEMP_START			# global temp state

## open the logger-file
if DO_SAVE_TO_FILE:
	os.makedirs(saveto_path)
	log.open(saveto_path)


## Error checking
if len(LEARNRATE_START)!= len(DBM_SHAPE)-1 and not DO_LOAD_FROM_FILE:
	log.out("DBM_SHAPE and LEARNRATE_START must have same length!")
	raise ValueError("DBM_SHAPE and LEARNRATE_START must have same length!")

if DO_LOAD_FROM_FILE and np.any(np.fromstring(PATHSUFFIX[26:].split("]")[0],sep=",",dtype=int) != DBM_SHAPE):
	log.out("Error: DBM Shape != Loaded Shape!")
	raise ValueError("DBM Shape != Loaded Shape!")
######### DBM #############################################################################################
DBM = DBM_class(	shape = DBM_SHAPE,
					liveplot = 0, 
					classification = 1,
			)

###########################################################################################################
#### Sessions ####
log.reset()
log.info(time_now)

# log.out("Not norming!")
DBM.pretrain()

if DO_TRAINING:
	log.start("DBM Train Session")
	

	with tf.Session() as sess:

		for run in range(N_EPOCHS_TRAIN):

			log.start("Run %i"%run)


			# start a train epoch 
			DBM.train(	train_data  = train_data,
						train_label = train_label if LOAD_MNIST else None,
						num_batches = N_BATCHES_TRAIN,
						cont        = run)

			# test session while training
			if run!=N_EPOCHS_TRAIN-1 and run%TEST_EVERY_EPOCH==0:
				# wrong_classified_id = np.loadtxt("wrongs.txt").astype(np.int)
				# DBM.test(train_data[:1000], train_label[:1000], 50, 10)

				DBM.test(test_data, test_label if LOAD_MNIST else None,
						N               = 20,  # sample ist aus random werten, also mindestens 2 sample machen 
						create_conf_mat = 0,
						temp_start      = temp,
						temp_end        = temp
						)

				log.out("Creating Backup of Parameters")
				DBM.backup_params()

				DBM.test(train_data[0:10000], train_label[0:10000] if LOAD_MNIST else None,
						N               = 20,  # sample ist aus random werten, also mindestens 2 sample machen 
						create_conf_mat = 0,
						temp_start      = temp,
						temp_end        = temp,
						using_train_data = True,
						)



			log.end()

	DBM.train_time=log.end()
	log.reset()

# last test session
if DO_TESTING:
	with tf.Session() as sess:
		DBM.test(test_data, test_label if LOAD_MNIST else None,
				N               = 50,  # sample ist aus random werten, also mindestens 2 sample machen 
				create_conf_mat = 1,
				temp_start      = temp,
				temp_end        = temp)

if DO_GEN_IMAGES:
	with tf.Session() as sess:
		log.start("Generation Session")

		if DO_LOAD_FROM_FILE and not DO_TRAINING:
			DBM.load_from_file(workdir+"/data/"+PATHSUFFIX,override_params=1)		

		nn                = 10		# grid with nn^2 plots (has to stay 10 for now)
		DBM.batchsize     = nn**2
		
		DBM.graph_init("gibbs")
		DBM.import_()

		label_clamp = np.zeros([DBM.batchsize,10])
		# set the correct label value
		for j in range(10):
			label_clamp[10*j:10*j+10,j] = 1

		generated_img = DBM.gibbs_sampling(label_clamp, 
							1000, 
							temp, temp, 
							100, 100,
							mode     = "generate",
							subspace = [],
							liveplot = 0)


		## plot the images (visible layers)
		fig,ax = plt.subplots(nn,nn)
		m = 0
		for i in range(nn):
			for j in range(nn):			
				ax[i,j].matshow(generated_img[m].reshape(28, 28))
				ax[i,j].set_xticks([])
				ax[i,j].set_yticks([])
				ax[i,j].grid(False)
				m += 1
		save_fig(saveto_path+"/generated_img.png",DO_SAVE_TO_FILE)

		## plot the enrgies
		fig_en,ax_en = plt.subplots(nn,nn,figsize=(13,10),sharex="col",sharey="row")
		m=0
		for i in range(nn):
			for j in range(nn):				
				ax_en[i,j].plot(DBM.energy_generate[:,m])				
				m += 1
				ax_en[-1,j].set_xlabel("Timestep t")
			ax_en[i,0].set_ylabel("Energy")
		plt.tight_layout()
		save_fig(saveto_path+"/generated_energy.png",DO_SAVE_TO_FILE)
		log.end()

if DO_CONTEXT:
	with tf.Session() as sess:
		log.start("Context Session")

		if DO_LOAD_FROM_FILE and not DO_TRAINING:
			DBM.load_from_file(workdir+"/data/"+PATHSUFFIX,override_params=1)

		subspace = [0,1,2,3,4]
		log.out("Subspace: ", subspace)

	
		# loop through images from all wrong classsified images and find al images that are <5 
		index_for_number_gibbs=[]
		for i in range(10000):
			
			## find the digit that was presented
			digit=np.where(test_label[i])[0][0] 		
			
			## set desired digit range
			if digit in subspace:
				index_for_number_gibbs.append(i)
		
		log.info("Found %i Images"%len(index_for_number_gibbs))


		# create graph 
		DBM.batchsize=len(index_for_number_gibbs)	
		if DBM.batchsize==0:
			raise ValueError("No Images found")

		DBM.graph_init("gibbs")
		DBM.import_()


		# calculte h2 firerates over all gibbs_steps 
		log.start("Sampling data")
		h2_no_context = DBM.gibbs_sampling(test_data[index_for_number_gibbs[:]], 100, 
							temp , temp, 
							999, 999,
							mode     = "context",
							subspace = "all",
							liveplot = 0)

		# # with context
		h2_context = DBM.gibbs_sampling(test_data[index_for_number_gibbs[:]], 100, 
							temp , temp, 
							999, 999,
							mode     = "context",
							subspace = subspace,
							liveplot = 0)
		# log.end()
		# DBM.export()

		# append h2 activity to array, but only the unit that corresponst to the given digit picture
		desired_digits_c  = []
		desired_digits_nc = []
		wrong_digits_c    = []
		wrong_digits_nc   = []

		correct_maxis_c    = []
		correct_maxis_nc   = []
		incorrect_maxis_c  = []
		incorrect_maxis_nc = []

		wrongs_outside_subspace_c = 0
		wrongs_outside_subspace_nc = 0

		hist_data    = np.zeros([10,1,10]).tolist()
		hist_data_nc = np.zeros([10,1,10]).tolist()

		for i,d in enumerate(index_for_number_gibbs):
			digit = np.where( test_label[d] == 1 )[0][0]
			
			hist_data[digit].append( h2_context[i].tolist() )
			hist_data_nc[digit].append( h2_no_context[i].tolist() )

			### count how many got right (with context) 
			## but only count the labels within subspace
			maxi_c         = h2_context[i][subspace[:]].max()
			maxi_all_pos_c = np.where(h2_context[i] == h2_context[i].max())[0][0]
			max_pos_c      = np.where(h2_context[i] == maxi_c)[0][0]
			if max_pos_c   == digit:
				correct_maxis_c.append(maxi_c)
			else:
				if maxi_all_pos_c  not  in  subspace:
					wrongs_outside_subspace_c += 1
				incorrect_maxis_c.append(maxi_c)

			### count how many got right (no context) 
			## but only count the labels within subspace
			maxi_nc     = h2_no_context[i][subspace[:]].max()
			maxi_all_pos_nc = np.where(h2_no_context[i]==h2_no_context[i].max())[0][0]
			max_pos_nc  = np.where(h2_no_context[i] == maxi_nc)[0][0]			
			if max_pos_nc == digit:
				correct_maxis_nc.append(maxi_nc)
			else:
				if maxi_all_pos_nc  not in  subspace:
					wrongs_outside_subspace_nc += 1
				incorrect_maxis_nc.append(maxi_nc)

			desired_digits_c.append(h2_context[i,digit])
			desired_digits_nc.append(h2_no_context[i,digit])

			wrong_digits_c.append(np.mean(h2_context[i,digit+1:])+np.mean(h2_context[i,:digit]))
			wrong_digits_nc.append(np.mean(h2_no_context[i,digit+1:])+np.mean(h2_context[i,:digit]))



		log.info("Inorrect Context:" , len(incorrect_maxis_c),"/",round(100*len(incorrect_maxis_c)/float(len(index_for_number_gibbs)),2),"%")
		log.info("Inorrect No Context:" , len(incorrect_maxis_nc),"/",round(100*len(incorrect_maxis_nc)/float(len(index_for_number_gibbs)),2),"%")
		log.info("Diff:     ",len(incorrect_maxis_nc)-len(incorrect_maxis_c))
		log.info("Outside subspace (c/nc):",wrongs_outside_subspace_c,",", wrongs_outside_subspace_nc)
		log.out("Means: Correct // Wrong (c/nc): \n \t \t ", 	round(np.mean(correct_maxis_c),4),
																round(np.mean(correct_maxis_nc),4), "//",
																round(np.mean(incorrect_maxis_c),4),
																round(np.mean(incorrect_maxis_nc),4)
				)
		## save to file
		if DO_SAVE_TO_FILE:
			file_gs = open(saveto_path+"/context_results.txt","w")
			file_gs.write("Found images: %i"%len(index_for_number_gibbs)+"\n")
			file_gs.write("Inorrect Context: " + str(len(incorrect_maxis_c))+"/"+str(round(100*len(incorrect_maxis_c)/float(len(index_for_number_gibbs)),2))+"%"+"\n")
			file_gs.write("Inorrect No Context: " + str(len(incorrect_maxis_nc))+"/"+str(round(100*len(incorrect_maxis_nc)/float(len(index_for_number_gibbs)),2))+"%"+"\n")
			file_gs.write("Diff: "+str(len(incorrect_maxis_nc)-len(incorrect_maxis_c))+"\n")
			file_gs.write("Outside subspace (c/nc): "+str(wrongs_outside_subspace_c)+","+ str(wrongs_outside_subspace_nc))
			# save hist data
			os.makedirs(saveto_path+"/context_hist")
			for cl in range(10):
				np.savetxt(saveto_path+"/context_hist/hist_data_c_digit_%i"%cl,np.array(hist_data[cl]))
				np.savetxt(saveto_path+"/context_hist/hist_data_nc_digit_%i"%cl,np.array(hist_data_nc[cl]))
			file_gs.close()


		# # calc how many digits got badly classified under a threshold 
		# wrong_class_nc = [np.sum(np.array(desired_digits_nc)[:]<i) for i in np.linspace(0,1,1000)]
		# wrong_class_c  = [np.sum(np.array(desired_digits_c)[:]<i)  for i in np.linspace(0,1,1000)]

		# wrong_class_nc2 = [np.sum(np.array(wrong_digits_nc)[:]>i) for i in np.linspace(0,1,1000)]
		# wrong_class_c2  = [np.sum(np.array(wrong_digits_c)[:]>i)  for i in np.linspace(0,1,1000)]


		# plt.figure()
		# plt.plot(np.linspace(0,1,1000),wrong_class_c,"-",label="With Context")
		# plt.plot(np.linspace(0,1,1000),wrong_class_nc,"-",label="Without Context")
		# plt.plot(np.linspace(0,1,1000),wrong_class_c2,"-",label="With Context / Mean")
		# plt.plot(np.linspace(0,1,1000),wrong_class_nc2,"-",label="Without Context / Mean")
		# plt.title("How many digits got classified below Threshold")
		# plt.xlabel("Threshold")
		# plt.ylabel("Number of Digits")
		# plt.legend(loc="best")

		# plot time series
		fig,ax = plt.subplots(1,3,figsize=(13,4));
		for i in range(DBM.n_layers-1):
			color = next(ax[0]._get_lines.prop_cycler)['color'];
			# color="r"
			ax[0].plot(DBM.activity_nc[i],"--",color = color)
			ax[0].plot(DBM.activity_c[i],"-",color   = color)
			label_str = get_layer_label(DBM.n_layers, i+1)
			ax[0].plot(0,0,color=color,label=label_str)
		ax[0].plot(1,1,"k--",label="No Context")
		ax[0].plot(1,1,"k-",label="with Context")
		ax[0].legend(loc="upper right")
		ax[0].set_ylabel("Active Neurons in %")
		ax[0].set_xlabel("Timestep")

		for i in range(DBM.n_layers-1):
			color = next(ax[1]._get_lines.prop_cycler)['color'];
			# color="r"
			
			ax[1].plot(DBM.layer_diff_gibbs_c[i,1:],"-",color=color)
			ax[1].plot(DBM.layer_diff_gibbs_nc[i,1:],"--",color=color)
			label_str = get_layer_label(DBM.n_layers, i+1)
			ax[1].plot(0,0,color=color,label=label_str)
		ax[1].plot(0,0,"k--",label="No Context")
		ax[1].plot(0,0,"k-",label="with Context")
		ax[1].legend(loc="best")
		ax[1].set_ylabel("|Layer(t) - Layer(t-1)|")
		ax[1].set_xlabel("Timestep")

		ax[2].plot(DBM.class_error_gibbs_c,"-",color=color)
		ax[2].plot(DBM.class_error_gibbs_nc,"--",color=color)

		ax[2].plot(0,0,"k--",label="No Context")
		ax[2].plot(0,0,"k-",label="With Context")
		ax[2].legend(loc="best")
		ax[2].set_ylabel("Class Error")
		ax[2].set_xlabel("Timestep")
		# plt.subplots_adjust(bottom=None, right=0.73, left=0.1, top=None,
		# 	            wspace=None, hspace=None)
		plt.tight_layout()
		save_fig(saveto_path+"/context_time_series.pdf",DO_SAVE_TO_FILE)



		### plt histograms for each used digit
		fig,ax = plt.subplots(1,len(subspace),figsize=(12,7),sharey="row")
		for i,digit in enumerate(subspace):
			y_nc = np.mean(hist_data_nc[digit][1:],axis=0)
			y_c  = np.mean(hist_data[digit][1:],axis=0)
			for j in range(10):
				if y_nc[j]>y_c[j]:
					ax[i].bar(j,y_nc[j],color=[0.8,0.1,0.1],label="Without Context",linewidth=0.1,edgecolor="k")
					ax[i].bar(j,y_c[j],color=[0.1,0.7,0.1],label="With Context",linewidth=0.1,edgecolor="k")
				else:
					ax[i].bar(j,y_c[j],color=[0.1,0.7,0.1],label="With Context",linewidth=0.1,edgecolor="k")
					ax[i].bar(j,y_nc[j],color=[0.8,0.1,0.1],label="Without Context",linewidth=0.1,edgecolor="k")

			plt.legend(loc="center left",bbox_to_anchor = (1.0,0.5))
			ax[i].set_ylim([0,1])
			ax[0].set_ylabel("Mean Predicted Label")
			ax[i].set_title(str(digit))
			ax[i].set_xticks(range(10))
		plt.subplots_adjust(bottom=None, right=0.84, left=0.1, top=None,
	            wspace=None, hspace=None)

		save_fig(saveto_path+"/context_hists.pdf",DO_SAVE_TO_FILE)



		# plot the variance of the layers for c/nc normed to nc and the firerates as hist
		log.out("Plotting variance diff c/nc")
		plt.figure()
		layer_str = [""]*DBM.n_layers
		for i in range(DBM.n_layers):
			diff = DBM.layer_diversity_c[i]/DBM.layer_diversity_nc[i]
			layer_str[i] = get_layer_label(DBM.n_layers,i,short=True)
			plt.bar(i,diff)
		plt.xticks(range(DBM.n_layers),layer_str)
		plt.xlabel("Layer")
		plt.ylabel("Diversity")
		save_fig(saveto_path+"/context_l_diversity.pdf",DO_SAVE_TO_FILE)
	



		# plot the unit variance c/nc
		fig,ax = plt.subplots(2,DBM.n_layers-2,figsize=(12,6))
		biggest_change_ind = [None]*(DBM.n_layers-2)
		for i in range(1,DBM.n_layers-1):
			delta_sigma = DBM.unit_diversity_c[i]-DBM.unit_diversity_nc[i]

			# magic_number = 0.0001234567 # if this number is in clean data it means +inf
			# delta_sigma = np.copy(delta_sigma)
			# delta_sigma[np.isnan(delta_sigma)] = 0 
			# delta_sigma[np.isinf(delta_sigma)] = magic_number
			biggest_change_ind[i-1] = np.where((delta_sigma > (delta_sigma.std()+delta_sigma.mean())) | (delta_sigma < (delta_sigma.mean()-delta_sigma.std())))[0]
			# print(np.mean(delta_sigma[np.isfinite(delta_sigma)]))
			

			ax[1,i-1].hist(delta_sigma,bins=30,	linewidth = 0.2, 
							edgecolor = "k")

			ax[0,i-1].hist(np.mean(DBM.firerates_c[i-1][:],0),bins=20,alpha=0.7,label = "With context",lw=0.2,edgecolor="k")
			ax[0,i-1].hist(np.mean(DBM.firerates_nc[i-1][:],0),bins=20,alpha=0.7,label = "Without context",lw=0.2,edgecolor="k")

			layer_str = get_layer_label(DBM.n_layers,i-1,short=True)
			# plt.colorbar(ax=ax[i-1],mappable=mapp)#,cbarlabel="$\sigma_%s^c/\sigma_%s^{nc}$"%(layer_str,layer_str))
			ax[1,i-1].set_xlabel(r"$\Delta \sigma_{%s}$"%layer_str[1:-1])
			ax[1,i-1].set_ylabel("N",style= "italic")
			ax[0,i-1].set_xlim([0,1])
			ax[0,i-1].set_xlabel(r"$<f>_{batch}$")
			ax[0,i-1].set_ylabel("N",style= "italic")
			ax[0,-1].legend(loc="best")

			# ax[i-1].xaxis.set_ticks_position("bottom")
			# ax[i-1].set_xticks([])
			# ax[i-1].set_yticks([])
		plt.tight_layout()
		save_fig(saveto_path+"/context_unit_div.pdf",DO_SAVE_TO_FILE)

		# count how many neurons got more active during context and how mch more

		# look at neuron hists for the neurons with biggest varianc change
		h_layer      = 2
		num_plots    = 5
		neuron_index = w_test#np.where((np.mean(DBM.firerates_nc[2],0)>0.01) & (np.mean(DBM.firerates_nc[2],0)<0.25) )[0]# range(200)#biggest_change_ind[h_layer]
		# max_len      = len(neuron_index)
		# if max_len>num_plots**2:
		# 	max_len = num_plots**2

		hists_c  = calc_neuron_hist(neuron_index, DBM.firerates_c[h_layer],  test_label[index_for_number_gibbs[:]], 0.9, len(subspace))
		hists_nc = calc_neuron_hist(neuron_index, DBM.firerates_nc[h_layer], test_label[index_for_number_gibbs[:]], 0.9, len(subspace))
		hists_c = np.array(hists_c)
		hists_nc = np.array(hists_nc)
		# diffs = []
		# for j in range(len(neuron_index)):
		# 	diffs.append(np.mean(np.abs(hists_c[j]-hists_nc[j])))
		# diffs = np.array(diffs)
		# # get the 9 biggest diffs
		# where_max_diffs = np.where(diffs>sorted(diffs)[-10])[0]
		log.out("die mittelung aller hists (c/nc) ueber alle neurone plotten - evtl steigen die ja beim context (plot auch nur über subspace möglich)")
		fig,ax = plt.subplots(1)
		ax.bar(np.array(subspace)-0.25,np.mean(hists_c,0),color="g",width=0.5)
		ax.bar(np.array(subspace)+0.25,np.mean(hists_nc,0),color="r",width=0.5)
		ax.set_xticks(range(10))
		# for j in range(num_plots):
		# 	index = m#where_max_diffs[m]
		# 	ax[i,j].bar(subspace,hists_c[index],alpha=0.7,color=[0.0, 1, 0.1])
		# 	ax[i,j].bar(range(10),hists_nc[index],alpha=0.7,color=[1, 0.0, 0.0])
		# 	ax[i,j].set_xticks(range(10))
		# 	ax[i,j].set_title(str(index)+" | "+ str(diffs[index]))
		# 	m+=1
		log.out("neben die plots oder sonstwo die entsprechenden varianzen plotten, und wirklich mal sicherstellen das meine indexe noch richtig sind")

		# plt.tight_layout()

	log.end() #end session


if DO_NOISE_STAB:
	with tf.Session() as sess:
		plt.figure()
		my_pal=["#FF3045","#77d846","#466dd8","#ffa700","#48e8ff","#a431e5","#333333","#a5a5a5","#ecbdf9","#b1f6b6"]
		noise_h2_,v_noise_recon,v_noise=DBM.test_noise_stability(test_data[0:10], test_label[0:10],20)
		# with seaborn.color_palette(my_pal, 10):
		# 	for i in range(10):
		# 		plt.plot(smooth(noise_h2_[:,0,i],10),label=str(i))
		# 	plt.legend()
		fig,ax=plt.subplots(2,10,figsize=(10,4))
		for i in range(10):
			ax[0,i].matshow(v_noise[i].reshape(28,28))
			ax[1,i].matshow(v_noise_recon[i].reshape(28,28))
			ax[0,i].set_yticks([])
			ax[1,i].set_yticks([])

		plt.tight_layout(pad=0.0)


if DO_SAVE_TO_FILE:
	DBM.write_to_file()


####################################################################################################################################
#### Plot
# Plot the Weights, Errors and other informations
h1_shape = int(sqrt(DBM.SHAPE[1]))

log.out("Plotting...")

if DO_TRAINING:
	# plot w1 as image	
	fig=plt.figure(figsize=(9,9))
	map1=plt.matshow(tile(DBM.w_np[0]),cmap="gray",fignum=fig.number)
	plt.colorbar(map1)
	plt.grid(False)
	plt.title("W %i"%0)
	save_fig(saveto_path+"/weights_img.pdf", DO_SAVE_TO_FILE)

	# plot layer diversity
	plt.figure("Layer diversity")
	for i in range(DBM.n_layers):
		label_str = get_layer_label(DBM.n_layers, i)
		plt.plot(range(DBM.n_layers)[::10],smooth(np.array(DBM.layer_diversity_train)[::2,i],10),label=label_str,alpha=0.7)
		plt.legend()
	plt.xlabel("Update Number")
	plt.ylabel("Deviation")
	save_fig(saveto_path+"/layer_diversity.png", DO_SAVE_TO_FILE)	

	plt.figure("Errors")
	## train errors
	plt.plot(range(DBM.n_layers)[::10],DBM.recon_error_train[:],"-",label="Recon Error Train",alpha=0.8)

	if DBM.classification:
		plt.plodt(range(DBM.n_layers)[::10],DBM.class_error_train[:],"-",label="Class Error Train",alpha=0.8)
	## test errors
	# calc number of updates per epoch
	n_u_p_e = len(DBM.recon_error_train) / DBM.epochs
	x = np.array(DBM.save_dict["Test_Epoch"])*n_u_p_e
	plt.plot(x,DBM.save_dict["Recon_Error"],"o--",label="Recon Error Test")
	if DBM.classification:
		plt.plot(x,DBM.save_dict["Class_Error"],"o--",label="Class Error Test")
		plt.plot(x[:-1],DBM.save_dict["Class_Error_Train_Data"],"o--",label="Class Error Test\nTrain Data")
	plt.legend(loc="best")
	plt.xlabel("Update Number")
	plt.ylabel("Mean Square Error")
	save_fig(saveto_path+"/errors.png", DO_SAVE_TO_FILE)


	# plot all other weights as hists
	log.out("Plotting Weights histograms")
	n_weights = DBM.n_layers-1
	fig,ax    = plt.subplots(n_weights,1,figsize=(8,10),sharex="col")
	for i in range(n_weights):
		if n_weights>1:
			seaborn.distplot(DBM.w_np[i].flatten(),rug=False,bins=60,ax=ax[i],label="After Training")
			ylim = ax[i].get_ylim()
			ax[i].axvline(DBM.w_np[i].max(),0,0.2, linestyle="-", color="k")
			ax[i].axvline(DBM.w_np[i].min(),0,0.2, linestyle="-", color="k")
			# ax[i].hist((DBM.w_np[i]).flatten(),bins=60,alpha=0.5,label="After Training")
			ax[i].set_title("W %i"%i)
			# ax[i].set_ylim(-ylim[1]/5,ylim[1])
			ax[i].legend()
		else:
			ax.hist((DBM.w_np[i]).flatten(),bins=60,alpha=0.5,label="After Training")
			ax.set_title("W %i"%i)
			ax.legend()

		try:
			seaborn.distplot(DBM.w_np_old[i].flatten(),rug=False,bins=60,ax=ax[i],label="Before Training",color="r")
			# ax[i].hist((DBM.w_np_old[i]).flatten(),color="r",bins=60,alpha=0.5,label="Before Training")
		except:
			pass
	plt.tight_layout()
	save_fig(saveto_path+"/weights_hist.pdf", DO_SAVE_TO_FILE)
	try:
		# plot change in w1 
		fig=plt.figure(figsize=(9,9))
		plt.matshow(tile(DBM.w_np[0]-DBM.w_np_old[0]),fignum=fig.number)
		plt.colorbar()
		plt.title("Change in W1")
		save_fig(saveto_path+"/weights_change.pdf", DO_SAVE_TO_FILE)
	except:
		plt.close(fig)


	# plot freerun diffs
	fig,ax = plt.subplots(1,1)
	for i in range(DBM.n_layers):
		l_str = get_layer_label(DBM.n_layers,i)
		ax.semilogy(range(DBM.update)[::10], DBM.freerun_diff_train[:,i], label = l_str)
	plt.ylabel("log("+r"$\Delta L$"+")")
	plt.legend(ncol = 2, loc = "best")
	plt.xlabel("Update")


	# plot train data (temp, diffs, learnrate, ..)
	fig,ax = plt.subplots(4,1,sharex="col",figsize=(8,8))

	ax[0].plot(DBM.save_dict["Temperature"],label="Temperature")
	ax[0].legend(loc="center left",bbox_to_anchor = (1.0,0.5))

	ax[0].set_ylabel("Temperature")

	ax[1].plot(DBM.save_dict["Learnrate"],label="Learnrate")
	ax[1].legend(loc="center left",bbox_to_anchor = (1.0,0.5))
	ax[1].set_ylabel("Learnrate")

	ax[2].set_ylabel("Mean")
	for i in range(len(DBM.SHAPE)-1):
		ax[2].plot(DBM.save_dict["W_mean_%i"%i],label="W %i"%i)
	ax[2].legend(loc="center left",bbox_to_anchor = (1.0,0.5))

	ax[3].set_ylabel("Diff")
	if DBM.epochs>2:	
		for i in range(len(DBM.SHAPE)-1):
			ax[3].semilogy(DBM.save_dict["W_diff_%i"%i][1:],label = "W %i"%i)
		for i in range(len(DBM.SHAPE)):
			ax[3].semilogy(DBM.save_dict["Bias_diff_%i"%i][1:],label = "Bias %i"%i)

	ax[3].legend(loc="center left",bbox_to_anchor = (1.0,0.5))

	ax[-1].set_xlabel("Epoch")

	plt.subplots_adjust(bottom=None, right=0.73, left=0.2, top=None,
	            wspace=None, hspace=None)

	save_fig(saveto_path+"/learnr-temp.pdf", DO_SAVE_TO_FILE)



	
	plt.figure("Layer_activiations_train_run")
	for i in range(DBM.n_layers):
		label_str = get_layer_label(DBM.n_layers, i+1)
		plt.plot(range(DBM.update)[::10],np.array(DBM.layer_act_train)[:,i],label = label_str)
	plt.legend()
	plt.xlabel("Update Number")
	plt.ylabel("Train Layer Activity in %")
	save_fig(saveto_path+"/layer_act_train.png", DO_SAVE_TO_FILE)

if LOAD_MNIST and DO_TESTING:
	
	## plot layer activities % test run
	plt.figure("Layer_activiations_test_run")
	for i in range(DBM.n_layers):
		label_str = get_layer_label(DBM.n_layers, i)
		plt.plot(DBM.layer_act_test[:,i],label=label_str)
	plt.legend()
	plt.xlabel("timestep")
	plt.ylabel("Test Layer Activity in %")
	save_fig(saveto_path+"/layer_act_test.pdf", DO_SAVE_TO_FILE)

	# plot the layer diversity after test run
	plt.figure("Layer stddeviation across test batch")
	plt.bar(range(DBM.n_layers),DBM.layer_diversity_test)
	plt.ylabel("Standard Deviation")
	plt.xlabel("Layer")
	plt.xticks(range(DBM.n_layers),[get_layer_label(DBM.n_layers, i ,short=True) for i in range((DBM.n_layers))])

	# plot l_input_test
	fig,ax = plt.subplots(1,1)

	color_m = 1

	m = 0
	max_y = 0
	for i in range(DBM.n_layers):
		color = next(ax._get_lines.prop_cycler)['color'];
		# color=[0.1,0.3,0.8]

		for direc in range(2):
			if direc == 0:
				versch = -0.125
				color_m = 1
			else:
				versch = +0.125
				color_m = 1.2

			max_y_ = np.max(np.mean(np.abs(DBM.l_input_test[i][direc])))
			if max_y_>max_y:
				max_y = max_y_
			ax.bar(m+versch, np.mean(np.abs(DBM.l_input_test[i][direc])),width = 0.25,
				color = np.multiply(color,color_m), linewidth = 0.2, edgecolor = "k",
				yerr  = np.mean(np.abs(DBM.l_var_test[i][direc]))
				)
			ax.set_ylabel("Mean Input")
		
		m+=1
		ax.set_xlabel("Layer")
		ax.set_ylim(0,max_y*1.2)
	save_fig(saveto_path+"/total_layer_input.pdf", DO_SAVE_TO_FILE)		
		

	# plot l_input_test as hist over all units
	fig,ax = plt.subplots(DBM.n_layers,1,figsize=(8,10),sharex="col")
	for i in range(DBM.n_layers):
		# max_x = 0
		ax_index = -(i+1)
		for direc in range(2):
			color = next(ax[i]._get_lines.prop_cycler)['color'];
			# color="r"
			label = "bottom up" if direc == 0 else "top down"
			data = np.array(DBM.hist_input_test[i][direc]).flatten()
			
			try:

				# max_x_ = data.max()
				# if max_x_>max_x:
				# 	max_x = max_x_

				y,x,_ = ax[ax_index].hist(data,
					bins      = 50,
					label     = label,
					color     = color,
					linewidth = 0.2, 
					edgecolor = "k",
					alpha     = 0.8,
					weights   = np.zeros_like(data)+1/data.size
					)

				
			except:
				pass

		# ax[i].set_ytick(ax[i].get_yticks())
		
		ax[ax_index].set_ylabel(r"$N/N_0$")
		ax[ax_index].set_title(get_layer_label(DBM.n_layers,i,short=True))
		ax[ax_index].legend()
	ax[-1].set_xlabel("Input Strength")
	plt.tight_layout()
	save_fig(saveto_path+"/layer_input_hist.pdf", DO_SAVE_TO_FILE)

	# plot timeseries of every neuron while testrun (clamped v)
	# layer_save_test has shape : [time][layer][image][neuron]
	k = 0 #which example image to pick
	if not os.path.isdir(saveto_path+"/timeseries_testrun"):
		os.makedirs(saveto_path+"/timeseries_testrun")
	for layer in range(1,DBM.n_layers):
	# 	timeseries = []
	# 	timeseries_average = []
	# 	for i in range(len(DBM.layer_save_test)):
	# 		timeseries.append(DBM.layer_save_test[i][layer][k])
	# 		timeseries_average.append(np.mean(DBM.layer_save_test[i][layer],0))
		# plot for image k

		plt.matshow(DBM.layer_save_test[layer][:,k])
		plt.xlabel("Time "+r"$t$")
		plt.ylabel("Unit "+r"$i$")
		save_fig(saveto_path+"/timeseries_testrun/timeseries_1image_layer_%i.png"%(layer+1),DO_SAVE_TO_FILE)
		
		# plt the average over all test images
		plt.matshow(np.mean(DBM.layer_save_test[layer][:,:],1))
		plt.xlabel("Time "+r"$t$")
		plt.ylabel("Unit "+r"$i$")
		
		save_fig(saveto_path+"/timeseries_testrun/timeseries_av_layer_%i.png"%(layer+1),DO_SAVE_TO_FILE)


	# plot some samples from the testdata 
	fig3,ax3 = plt.subplots(len(DBM.SHAPE)+1,13,figsize=(16,5),sharey="row")
	for i in range(13):
		# plot the input
		ax3[0][i].matshow(test_data[i:i+1].reshape(int(sqrt(DBM.SHAPE[0])),int(sqrt(DBM.SHAPE[0]))))
		ax3[0][i].set_yticks([])
		ax3[0][i].set_xticks([])
		# plot the reconstructed image		
		ax3[1][i].matshow(np.mean(DBM.layer_save_test[0][-20:,i],0).reshape(int(sqrt(DBM.SHAPE[0])),int(sqrt(DBM.SHAPE[0]))))
		ax3[1][i].set_yticks([])
		ax3[1][i].set_xticks([])
		
		#plot hidden layer
		for layer in range(1,len(DBM.SHAPE)-1):
			try:
				ax3[layer+1][i].matshow(DBM.layer_save_test[layer][-1][i].reshape(int(sqrt(DBM.SHAPE[layer])),int(sqrt(DBM.SHAPE[layer]))))
				ax3[layer+1][i].set_yticks([])
				ax3[layer+1][i].set_xticks([])
			except:
				pass
		# plot the last layer 	
		if DBM.classification:	
			ax3[-1][i].bar(range(DBM.SHAPE[-1]),DBM.last_layer_save[i])
			ax3[-1][i].set_xticks(range(DBM.SHAPE[-1]))
			ax3[-1][i].set_ylim(0,1)
		else:
			ax3[-1][i].matshow(DBM.last_layer_save[i].reshape(int(sqrt(DBM.SHAPE[-1])),int(sqrt(DBM.SHAPE[-1]))))
			ax3[-1][i].set_xticks([])
			ax3[-1][i].set_yticks([])

		#plot the reconstructed layer h1
		# ax3[5][i].matshow(DBM.rec_h1[i:i+1].reshape(int(sqrt(DBM.SHAPE[1])),int(sqrt(DBM.SHAPE[1]))))
		# plt.matshow(random_recon.reshape(int(sqrt(DBM.SHAPE[0])),int(sqrt(DBM.SHAPE[0]))))
	plt.tight_layout(pad=0.0)
	save_fig(saveto_path+"/examples.pdf", DO_SAVE_TO_FILE)

	# plot only one digit
	fig3,ax3 = plt.subplots(len(DBM.SHAPE)+1,10,figsize=(16,5),sharey="row")
	m=0
	for i in index_for_number_test.astype(np.int)[8][0:10]:
		# plot the input
		ax3[0][m].matshow(test_data[i:i+1].reshape(int(sqrt(DBM.SHAPE[0])),int(sqrt(DBM.SHAPE[0]))))
		ax3[0][m].set_yticks([])
		ax3[0][m].set_xticks([])
		# plot the reconstructed image		
		ax3[1][m].matshow(np.mean(DBM.layer_save_test[0][-20:,i],0).reshape(int(sqrt(DBM.SHAPE[0])),int(sqrt(DBM.SHAPE[0]))))
		ax3[1][m].set_yticks([])
		ax3[1][m].set_xticks([])
		
		#plot hidden layer
		for layer in range(1,len(DBM.SHAPE)-1):
			try:
				ax3[layer+1][m].matshow(DBM.layer_save_test[layer][-1][i].reshape(int(sqrt(DBM.SHAPE[layer])),int(sqrt(DBM.SHAPE[layer]))))
				ax3[layer+1][m].set_yticks([])
				ax3[layer+1][m].set_xticks([])
			except:
				pass
		# plot the last layer 		
		if DBM.classification:
			ax3[-1][m].bar(range(DBM.SHAPE[-1]),DBM.last_layer_save[i])
			ax3[-1][m].set_xticks(range(DBM.SHAPE[-1]))
			ax3[-1][m].set_ylim(0,1)
		#plot the reconstructed layer h1
		# ax4[5][m].matshow(DBM.rec_h1[i:i+1].reshape(int(sqrt(DBM.SHAPE[1])),int(sqrt(DBM.SHAPE[1]))))
		# plt.matshow(random_recon.reshape(int(sqrt(DBM.SHAPE[0])),int(sqrt(DBM.SHAPE[0]))))
		m+=1
	plt.tight_layout(pad=0.0)
	save_fig(saveto_path+"/examples_one_digit.pdf", DO_SAVE_TO_FILE)


if DO_CONTEXT:
	# plot layer input with and without context
	fig,ax  = plt.subplots(1,1)
	color_m = 1
	m       = 0
	max_y   = 0
	for i in range(DBM.n_layers):
		color = next(ax._get_lines.prop_cycler)['color'];
		# color="r"

		for direc in range(2):
			if direc == 0:
				versch = -0.125
				color_m = 1
			else:
				versch = +0.125
				color_m = 1.2

			data_c     = np.mean(np.abs(DBM.l_input_c[i][direc]))
			data_nc     = np.mean(np.abs(DBM.l_input_nc[i][direc]))
			filename = "/context_total_l_input_diff.pdf"
			
			
			ax.bar(m+versch, data_c-data_nc,width = 0.25, #hatch="///",
				 linewidth = 0.2, edgecolor = "k",
			
				)
			ax.set_ylabel("Mean Input")
		
		m+=1
		ax.set_xlabel("Layer")
	save_fig(saveto_path+filename, DO_SAVE_TO_FILE)


	# plot input  hist for context und no context	
	for mode in range(2):
		fig,ax = plt.subplots(DBM.n_layers,1,figsize=(8,10))
		for i in range(DBM.n_layers):
			max_x = 0
			
			for direc in range(2):
				color = next(ax[i]._get_lines.prop_cycler)['color'];
				# color="r"
				label = "bottom up" if direc == 0 else "top down"
				if mode ==0:
					data = np.array(DBM.hist_input_c[i][direc]).flatten()
					filename = "/context_hist_input_c.pdf"
				else:
					data = np.array(DBM.hist_input_nc[i][direc]).flatten()
					filename = "/context_hist_input_nc.pdf"

				if i == DBM.n_layers-1 and direc == 0 and mode == 0:
					eps = 0.0001
					data = data[np.abs(data)>eps]
					label = label + "\n(neglected zeros)"
				try:

					max_x_ = data.max()
					if max_x_>max_x:
						max_x = max_x_

					y,x,_ = ax[i].hist(data,
						bins      = 60,
						label     = label,
						color     = color,
						linewidth = 0.2, 
						edgecolor = "k",
						alpha     = 0.8,
						weights   = np.zeros_like(data)+1/data.size
						)

					if max_x!=0:
						ax[i].set_xlim([-max_x*1.2,max_x*1.2])
					
				except:
					pass

			# ax[i].set_ytick(ax[i].get_yticks())
			
			ax[i].set_ylabel(r"$N/N_0$")
			ax[i].legend()
		ax[-1].set_xlabel("Input Strength")
		save_fig(saveto_path+filename,DO_SAVE_TO_FILE)

if DO_GEN_IMAGES:
	# plot timeseries of every neuron while generate (clamped label)
	# layer_save_generate has shape : [time][layer][image][neuron]
	k = 0 #which example image to pick
	if not os.path.isdir(saveto_path+"/timeseries_generated"):
		os.makedirs(saveto_path+"/timeseries_generated")
	for layer in range(DBM.n_layers-1):
		# timeseries = []
		# timeseries_average = []
		# for i in range(len(DBM.layer_save_generate)):
		# 	timeseries.append(DBM.layer_save_generate[i][layer][k])
		# 	timeseries_average.append(np.mean(DBM.layer_save_generate[i][layer],0))
		

		# plot for image k
		plt.matshow(DBM.layer_save_generate[layer][:,k])	
		plt.ylabel("Time "+r"$t$")
		plt.xlabel("Unit "+r"$i$")
		save_fig(saveto_path+"/timeseries_generated/timeseries_1image_layer_%i.png"%(layer+1),DO_SAVE_TO_FILE)
		
		# plt the average over all test images
		plt.matshow(np.mean(DBM.layer_save_generate[layer][:,:],1))
		plt.ylabel("Time "+r"$t$")
		plt.xlabel("Unit "+r"$i$")
		
		save_fig(saveto_path+"/timeseries_generated/timeseries_av_layer_%i.png"%(layer+1),DO_SAVE_TO_FILE)


log.close()
if DO_SHOW_PLOTS:
	plt.show()
else:
	plt.close()
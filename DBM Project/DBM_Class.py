import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from RBM import *
###############################################################################################################
### Class Deep BM
class DBM_class(object):
	"""
	Defines a deep Boltzmann machine.
	"""
	def type(self):
		return "DBM"
		
	def __init__(self, UserSettings, logger, workdir, saveto_path, liveplot):

		### load UserSettings into self
		for key in UserSettings:
			self.__dict__[key] = UserSettings[key]

		self.log = logger
		self.workdir = workdir
		self.data_dir = os.path.join(workdir, "data")
		self.saveto_path = saveto_path
		

		### "global" state vars
		# temperature of DBM
		self.temp          = self.TEMP_START;
		# learnrate of DBM
		self.learnrate     = self.LEARNRATE_START;
		# how many freerun steps in train
		self.freerun_steps = self.N_FREERUN_START;


		self.n_layers       = len(self.DBM_SHAPE)
		# if true will open a lifeplot of the weight matrix
		self.liveplot       = liveplot
		# contains the number of  neurons in a list from v layer to h1 to h2
		self.SHAPE          = self.DBM_SHAPE 
		# weather the machine uses a label layer
		self.classification = self.DO_CLASSIFICATION

		# weather or not the graph init function was called
		self.init_state     = 0
		# if export function was called or not
		self.exported       = 0
		# if tested function was called or not (while training)
		self.tested         = 0
		# counts train time in seconds
		self.train_time     = 0	
		# epoch counter
		self.epochs         = 0	
		# update counter
		self.update         = 0 	
		# offset for the update param if loaded from file (used in get temp function)
		self.update_off     = 0	
		# how many label layer the system has
		self.n_label_layer  = 1

		#### arrays to store train and test results
		# save update
		self.updates               = np.arange(0,self.N_BATCHES_TRAIN*self.N_EPOCHS_TRAIN,10)	
		# save reconstructon error for every batch
		self.recon_error_train     = np.zeros([self.N_BATCHES_TRAIN*self.N_EPOCHS_TRAIN//10])	
		# -"- train error -"-
		self.class_error_train     = np.zeros([self.N_BATCHES_TRAIN*self.N_EPOCHS_TRAIN//10])	
		# save layer variance across batch for every batch in train function
		self.layer_diversity_train = np.zeros([self.N_BATCHES_TRAIN*self.N_EPOCHS_TRAIN//10, self.n_layers])	
		# save how many units are active across one layer in % for every batch
		self.layer_act_train       = np.zeros([self.N_BATCHES_TRAIN*self.N_EPOCHS_TRAIN//10, self.n_layers])	
		# save how diffrerent the freerun vs the clamp run while training is
		self.freerun_diff_train    = np.zeros([self.N_BATCHES_TRAIN*self.N_EPOCHS_TRAIN//10, self.n_layers])
		# layer activities while training (does not get plottet)
		self.l_mean                = np.zeros([self.n_layers])
		# not used
		# self.n_particles = 0; 

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



		### self.log list where all constants are saved
		## append variables that change during training in the write_to_file function
		self.log_list =	UserSettings


		## setting random seed 
		
		# tf.set_random_seed(self.SEED)

		self.log.out("Creating RBMs")
		self.RBMs    = [None]*(self.n_layers-1)
		for i in range(len(self.RBMs)):
			if i == 0 and len(self.RBMs)>1:
				self.RBMs[i] = RBM(self.SHAPE[i],self.SHAPE[i+1], forw_mult= 1, back_mult = 1, learnrate = self.LEARNRATE_PRETRAIN, liveplot=0, temp = self.temp)
				self.log.out("2,1")
			elif i==len(self.RBMs)-1 and len(self.RBMs)>1:
				self.RBMs[i] = RBM(self.SHAPE[i],self.SHAPE[i+1], forw_mult= 1, back_mult = 1, learnrate = self.LEARNRATE_PRETRAIN, liveplot=0, temp = self.temp)
				self.log.out("1,2")
			else:
				if len(self.RBMs) == 1:
					self.RBMs[i] = RBM(self.SHAPE[i],self.SHAPE[i+1], forw_mult= 1, back_mult = 1, learnrate = self.LEARNRATE_PRETRAIN, liveplot=0, temp = self.temp)
					self.log.out("1,1")
				else:
					self.RBMs[i] = RBM(self.SHAPE[i],self.SHAPE[i+1], forw_mult= 1, back_mult = 1, learnrate = self.LEARNRATE_PRETRAIN, liveplot=0, temp = self.temp)
					self.log.out("2,2")

	def pretrain(self, train_data):
		""" this function will pretrain the RBMs and define a self.weights list where every
		weight will be stored in. This weights list can then be used to save to file and/or
		to be loaded into the DBM for further training.
		"""

		if self.DO_PRETRAINING:
			for rbm in self.RBMs:
				if rbm.liveplot:
					self.log.info("Liveplot is open!")
					fig,ax=plt.subplots(1,1,figsize=(15,10))
					break

		batchsize_pretrain = int(len(train_data)/self.N_BATCHES_PRETRAIN)

		with tf.Session() as sess:
			# train session - v has batchsize length
			self.log.start("Pretrain Session")


			#iterate through the RBMs , each iteration is a RBM
			if self.DO_PRETRAINING:
				sess.run(tf.global_variables_initializer())

				for RBM_i, RBM in enumerate(self.RBMs):
					self.log.start("Pretraining ",str(RBM_i+1)+".", "RBM")


					for epoch in range(self.N_EPOCHS_PRETRAIN[RBM_i]):

						self.log.start("Epoch:",epoch+1,"/",self.N_EPOCHS_PRETRAIN[RBM_i])

						for start, end in zip( range(0, len(train_data), batchsize_pretrain), range(batchsize_pretrain, len(train_data), batchsize_pretrain)):
							#### define a batch
							batch = train_data[start:end]
							# train the rbm
							w_i, error_i = RBM.train(sess,RBM_i,self.RBMs,batch)
							#### liveplot
							if RBM.liveplot and plt.fignum_exists(fig.number) and start%40==0:
								ax.cla()
								rbm_shape  = int(sqrt(RBM.visible_units))
								matrix_new = tile_raster_images(X=w_i.T, img_shape=(rbm_shape, rbm_shape), tile_shape=(10, 10), tile_spacing=(0,0))
								ax.matshow(matrix_new)
								plt.pause(0.00001)


						self.log.info("Learnrate:",round(self.LEARNRATE_PRETRAIN,4))
						self.log.info("error",round(error_i,4))
						self.log.end() #ending the epoch


					self.log.end() #ending training the rbm



				# define the weights
				self.weights  =  []
				for i in range(len(self.RBMs)):
					self.weights.append(self.RBMs[i].w.eval())

				if self.DO_SAVE_PRETRAINED:
					for i in range(len(self.weights)):
						np.savetxt(self.workdir+"/pretrain_data/"+"Pretrained-"+" %i "%i+str(time_now)+".txt", self.weights[i])
					self.log.out("Saved Pretrained under "+str(time_now))
			else:
				if not self.DO_LOAD_FROM_FILE:
					### load the pretrained weights
					self.weights=[]
					self.log.out("Loading Pretrained from file")
					for i in range(self.n_layers-1):
						self.weights.append(np.loadtxt(self.workdir+"/pretrain_data/"+"Pretrained-"+" %i "%i+self.PATHSUFFIX_PRETRAINED+".txt").astype(np.float32))
				else:
					### if loading from file is active the pretrained weights would get
					### reloaded anyway so directly load them here
					self.weights=[]
					self.log.out("Loading from file")
					for i in range(self.n_layers-1):
						self.weights.append(np.loadtxt(data_dir+"/"+self.PATHSUFFIX+"/"+"w%i.txt"%(i)).astype(np.float32))
			self.log.end()
			self.log.reset()

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

	def load_from_file(self, path, override_params=0):
		""" loads weights and biases from folder and sets
		variables like learnrate and temperature to the values
		that were used in the last epoch"""
		os.chdir(path)
		self.log.out("Loading data from:","...",path[-20:])

		self.w_np     = []
		self.w_np_old = []
		r = self.n_layers-1
		if self.type() == "DBM_attention":
			r = self.n_layers-1 + len(self.layers_to_connect)
		for i in range(r):
			self.w_np.append(np.loadtxt("w%i.txt"%(i)))
			self.w_np_old.append(self.w_np[i])  #save weights for later comparison

		self.bias_np = []
		for i in range(self.n_layers):
			self.bias_np.append(np.loadtxt("bias%i.txt"%(i)))
		if override_params:
			# try:
			self.log.out("Overriding Values from save")

			## read save dict and self.log file and save as dicts
			sd = read_csv("save_dict.csv")
			log_dict = self.read_logfile()


			try:
				l_string = sd["Learnrate"].values[[sd["Learnrate"].notnull()]]
				l_string[-1]=l_string[-1].replace("["," ")
				l_string[-1]=l_string[-1].replace("]"," ")
				l_ = np.fromstring(l_string[-1][1:-1],sep=" ")
			except:
				l_ = sd["Learnrate"].values[[sd["Learnrate"].notnull()]]
			t_ = sd["Temperature"].values[[sd["Temperature"].notnull()]]
			n_ = sd["Freerun_Steps"].values[[sd["Freerun_Steps"].notnull()]]
			train_epoch_ = sd["Train_Epoch"].values[[sd["Train_Epoch"].notnull()]]

			self.freerun_steps = n_[-1]
			self.temp          = t_[-1]
			self.learnrate     = l_[-1]
			self.epochs        = train_epoch_[-1]

			try:
				self.update_off = log_dict["Update"]
			except:
				self.log.info("No key 'update' in logfile found.")


			self.log.info("Epoch = ",self.epochs)
			self.log.info("l = ",self.learnrate)
			self.log.info("T = ",round(self.temp,5))
			self.log.info("N = ",self.freerun_steps)
			# except:
			# 	self.log.error("Error overriding: Could not find save_dict.csv")
		os.chdir(self.workdir)

	def import_(self, sess):
		""" setting up the graph and setting the weights and biases tf variables to the
		saved numpy arrays """
		self.log.out("loading numpy vars into graph")
		for i in range(len(self.w_np)):
			sess.run(self.w[i].assign(self.w_np[i]))
		for i in range(len(self.bias_np)):
			sess.run(self.bias[i].assign(self.bias_np[i]))

	def layer_input(self, layer_i):
		""" calculate input of layer layer_i
		layer_i :: for which layer
		returns :: input for the layer - which are the probabilites
		"""
		if layer_i == 0:
			w = self.w[layer_i];
			if self.USE_DROPOUT:
				w.assign(w * self.dropout_matrix[layer_i])
			_input_ = sigmoid(tf.matmul(self.layer[layer_i+1], w, transpose_b=True) + self.bias[layer_i], self.temp_tf)

		elif layer_i == self.n_layers-1:
			w = self.w[layer_i-1];
			if self.USE_DROPOUT:
				w.assign(w * self.dropout_matrix[layer_i-1])
			_input_ = sigmoid(tf.matmul(self.layer[layer_i-1],w) + self.bias[layer_i], self.temp_tf)

		else:
			w0 = self.w[layer_i-1];
			w1 = self.w[layer_i];
			if self.USE_DROPOUT:
				w0.assign(w0 * self.dropout_matrix[layer_i-1])
				w1.assign(w1 * self.dropout_matrix[layer_i])
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
						x - tf.random_uniform(tf.shape(x), seed = self.SEED)
					)
				)

	def glauber_step(self, clamp, temp, droprate, save_array, step, sess):
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
		# rnd_order = sorted(rnd_order,reverse=True)

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

	def get_learnrate(self, epoch, a, y_off):
		""" calculate the learnrate dependend on parameters
		epoch :: current epoch
		a :: slope
		y_off :: y offset -> start learningrate
		"""

		L = a / (float(a)/y_off+epoch)
		return L

	def get_temp(self, update, a, y_off,  t_min):
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
		""" used to calculate new N dynamically """
		# N = epoch*1
		# if N<4:
		# 	N = 4
		# if N>50:
		N = 4
		return int(N)

	def update_savedict(self,mode, sess):
		if mode=="training":
			# append all data to save_dict
			self.save_dict["Train_Epoch"].append(self.epochs)
			self.save_dict["Temperature"].append(self.temp)
			self.save_dict["Learnrate"].append(self.learnrate)
			self.save_dict["Freerun_Steps"].append(self.freerun_steps)

			for i in range(len(self.w)):
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
					if self.type() == "DBM_attention":
						self.save_dict["Context_Error"].append(self.context_error_test)
			self.save_dict["Recon_Error"].append(self.recon_error)
			self.save_dict["Test_Epoch"].append(self.epochs)


		if mode=="testing_with_train_data":
			if self.classification:
				self.save_dict["Class_Error_Train_Data"].append(self.class_error_test)

	def get_total_layer_input(self, sess):
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

	def get_units_input(self, sess):
		### unit input histogram measure
		hist_input = [[[],[]] for _ in range(self.n_layers)]
		for i in range(self.n_layers-1):
			left_input = sess.run(tf.matmul(self.layer[i],  self.w[i]))
			right_input = sess.run(tf.matmul(self.layer[i+1],self.w[i],transpose_b=True))

			hist_input[i+1][0].append(left_input)
			hist_input[i][1].append(right_input)
		return hist_input

	def graph_init(self, graph_mode, sess):
		""" sets the graph up and loads the pretrained weights in , these are given
		at class definition
		graph_mode  :: "training" if the graph is used in training - this will define more variables that are used in training - is slower
					:: "testing"  if the graph is used in testing - this will define less variables because they are not needed in testing
		"""
		self.log.out("Initializing graph")


		if graph_mode=="training":
			self.learnrate_tf = tf.placeholder(tf.float32,name="Learnrate")

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
		self.assign_l_zeros    = [None]*self.n_layers # assign op. (assigns random)
		self.layer_prob        = [None]*self.n_layers # calc prob for layer n
		self.layer_samp        = [None]*self.n_layers # take a sample from the prob
		self.update_l_s        = [None]*self.n_layers # assign op. for calculated samples
		self.update_l_p        = [None]*self.n_layers # assign op. for calculated probs
		self.layer_activities  = [None]*self.n_layers # calc for layer activieties (mean over batch)
		self.layer_energy      = [None]*(self.n_layers-1)
		self.unit_diversity    = [None]*self.n_layers # measure how diverse each unit is in the batch
		self.layer_diversity   = [None]*self.n_layers # measure how diverse each layer is in the batch
		self.layer_variance    = [None]*self.n_layers # use this after the layer diversity calc to get variance
		self.freerun_diff      = [None]*self.n_layers # calculates mean(abs(layer_save (clamped run) - layer))
														# if called after freerunnning it tells the mean difference between freeerun and clamped run


		### layer vars
		for i in range(len(self.layer)):
			self.layer[i]      = tf.Variable(tf.random_uniform([self.batchsize,self.SHAPE[i]], minval=-1e-3, maxval=1e-3, seed = self.SEED), name="Layer_%i"%i)
			self.layer_save[i] = tf.Variable(tf.random_uniform([self.batchsize,self.SHAPE[i]], minval=-1e-3, maxval=1e-3, seed = self.SEED), name="Layer_save_%i"%i)
			# self.layer_particles[i] = tf.Variable(tf.zeros([self.batchsize,self.SHAPE[i]]),name="Layer_particle_%i"%i)
			self.layer_ph[i]   = tf.placeholder(tf.float32, [self.batchsize,self.SHAPE[i]], name="layer_%i_PH"%i)

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
				self.update_w[i]        = self.w[i].assign_add(self.learnrate*self.temp/self.batchsize*self.CD[i])
				self.w_mean_[i]         = tf.Variable(tf.zeros([self.N_EPOCHS_TRAIN]))
				self.len_w[i]           = tf.sqrt(tf.reduce_sum(tf.square(self.w[i])))

				## old norm approach
				self.do_norm_w[i] = self.w[i].assign(self.w[i]/self.len_w[i])

				# droprate is constant- used for training
				self.dropout_matrix[i]  = tf.round(tf.clip_by_value(tf.random_uniform(tf.shape(self.w[i]), seed = self.SEED)*self.DROPOUT_RATE,0,1))
			else:
				# droprate is variable and can be decreased over time
				self.dropout_matrix[i] = tf.round(tf.clip_by_value(tf.random_uniform(tf.shape(self.w[i]), seed = self.SEED)*self.droprate_tf,0,1))
		

		### append extra connections from context layer 
		if self.type() == "DBM_attention":
			# # array of weights connecting to the context layer (index i will show connection between layer[i] and context layer)
			for i,l in enumerate(self.layers_to_connect):
				index = i+len(self.w) # index continues at length of DBM weights 
				f = self.SHAPE[l] # from
				t = self.SHAPE[-1] # to

				## context weights and calculations
				self.w.append(tf.Variable(tf.zeros([f, t]), name = "context_weights"))

				if graph_mode=="training":
					self.pos_grad.append(			tf.Variable(tf.zeros([f, t])))
					self.neg_grad.append(			tf.Variable(tf.zeros([f, t])))
					self.update_pos_grad.append(	self.pos_grad[index].assign(tf.matmul(self.layer[l], self.layer[-1], transpose_a = True)))
					self.update_neg_grad.append(	self.neg_grad[index].assign(tf.matmul(self.layer[l], self.layer[-1], transpose_a = True)))
					self.CD.append(					self.pos_grad[index] - self.neg_grad[index])
					self.CD_abs_mean.append(		tf.reduce_mean(tf.abs(self.CD[index])))
					self.update_w.append(			self.w[index].assign_add(self.learnrate*self.temp/self.batchsize*self.CD[index]))
					self.w_mean_.append(			tf.Variable(tf.zeros([self.N_EPOCHS_TRAIN])))
					self.len_w.append(				tf.sqrt(tf.reduce_sum(tf.square(self.w[index]))))

					## old norm approach
					self.do_norm_w.append(self.w[index].assign(self.w[index]/self.len_w[index]))

					# droprate is constant- used for training
					self.dropout_matrix.append( tf.round(tf.clip_by_value(tf.random_uniform(tf.shape(self.w[index]), seed = self.SEED)*self.DROPOUT_RATE,0,1)))
				else:
					# droprate is variable and can be decreased over time
					self.dropout_matrix.append( tf.round(tf.clip_by_value(tf.random_uniform(tf.shape(self.w[index]), seed = self.SEED)*self.droprate_tf,0,1)))



		### bias calculations and assignments
		for i in range(len(self.bias)):
			self.bias[i] = tf.Variable(tf.zeros([self.SHAPE[i]]),name="Bias%i"%i)
			if graph_mode == "training":
				self.update_bias[i] = self.bias[i].assign_add(self.learnrate*self.temp*tf.reduce_mean(tf.subtract(self.layer_save[i],self.layer[i]),0))
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
			self.assign_l_zeros[i]     = self.layer[i].assign(tf.zeros([self.batchsize,self.SHAPE[i]]))
			self.layer_prob[i]        = self.layer_input(i)
			self.layer_samp[i]        = self.sample(self.layer_prob[i])
			self.update_l_p[i]        = self.layer[i].assign(self.layer_prob[i])
			self.layer_activities[i]  = tf.reduce_sum(self.layer[i])/(self.batchsize*self.SHAPE[i])*100
			self.unit_diversity[i]    = tf.sqrt(tf.reduce_mean(tf.square(self.layer[i] - tf.reduce_mean(self.layer[i], axis=0)),axis=0))
			self.layer_diversity[i]   = tf.reduce_mean(self.unit_diversity[i])
			self.layer_variance[i]    = tf.nn.moments(self.unit_diversity[i],0)[1]
			self.freerun_diff[i]      = tf.reduce_mean(tf.abs(self.layer_save[i]-self.layer[i]))

		for i in range(len(self.layer)-1):
			if i <len(self.layer)-2:
				self.layer_energy[i] = tf.einsum("ij,ij->i",self.layer[i+1], tf.matmul(self.layer[i],self.w[i]))+tf.reduce_sum(self.layer[i]*self.bias[i],1)
			else:
				self.layer_energy[i] = tf.einsum("ij,ij->i",self.layer[i+1], tf.matmul(self.layer[i],self.w[i]))+tf.reduce_sum(self.layer[i]*self.bias[i],1)+tf.reduce_sum(self.layer[i+1]*self.bias[i+1],1)
			self.update_l_s[i]   = self.layer[i].assign(self.layer_samp[i])
		self.update_l_s[-1] = self.layer[-1].assign(self.layer_prob[-1])


		### Error and stuff
		self.error       = tf.reduce_mean(tf.square(self.layer_ph[0]-self.layer[0]))
		self.class_error = tf.reduce_mean(tf.square(self.layer_ph[-1]-self.layer[-1]))

		self.free_energy = -tf.reduce_sum(tf.log(1+tf.exp(tf.matmul(self.layer[0],self.w[0])+self.bias[1])))

		self.energy = -tf.add_n([self.layer_energy[i] for i in range(len(self.layer_energy))])



		sess.run(tf.global_variables_initializer())
		self.init_state = 1

	def train(self, train_data, test_data, train_label = None, test_label = None):
		""" wrapper for the old _train_() function """
		self.train_data = train_data

		if self.DO_TRAINING:
			self.log.start("DBM Train Session")


			with tf.Session() as sess:

				for run in range(self.N_EPOCHS_TRAIN):

					self.log.start("Run %i"%run)


					### start a train epoch
					self._train_(
						train_data  = train_data,
						train_label = train_label,
						num_batches = self.N_BATCHES_TRAIN,
						cont        = run,
						sess	 	= sess,
					)

					### test session while training
					if run!=self.N_EPOCHS_TRAIN-1 and run%self.TEST_EVERY_EPOCH==0:
						self._test_(	test_data, 
										test_label,
										N               = 30,  # sample beginnt aus random werten, also mindestens 2 sample machen
										create_conf_mat = 0,
										temp_start      = self.temp,
										temp_end        = self.temp,
										sess 			= sess,
						)

					### backup params
						# self.log.out("Creating Backup of Parameters")
						# self.backup_params()




					self.log.end()

			self.train_time=self.log.end()
			self.log.reset()

	def _train_(self, train_data, train_label, num_batches, cont, sess):
		"""
		(Old train function without session definition. Use new wrapper function train() instead)
		training the DBM with given h2 as labels and v as input images
		train_data  :: images
		train_label :: corresponding label
		num_batches :: how many batches
		"""
		######## init all vars for training
		self.batchsize = int(len(train_data)/num_batches)
		self.num_of_updates = self.N_EPOCHS_TRAIN*num_batches


		# number of clamped sample steps
		if self.n_layers <=3 and self.classification==1:
			M = 2
		else:
			M = 20





		### free energy
		# self.F=[]
		# self.F_test=[]

		if self.DO_LOAD_FROM_FILE and not cont:
			# load data from the file
			self.load_from_file(self.workdir+"/data/"+self.PATHSUFFIX,override_params=1)
			self.graph_init("training",sess)
			self.import_(sess)

		# if no files loaded then init the graph with pretrained vars
		if self.init_state==0:
			self.graph_init("training",sess)

		if cont and self.tested:
			self.graph_init("training",sess)
			self.import_(sess)
			self.tested = 0
			self.learnrate = self.get_learnrate(self.epochs, self.LEARNRATE_SLOPE, self.LEARNRATE_START)


		if self.liveplot:
			self.log.info("Liveplot is on!")
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
		self.log.info("Batchsize:",self.batchsize,"N_Updates:",self.num_of_updates)

		self.log.start("Deep BM Epoch:",self.epochs+1,"/",self.N_EPOCHS_TRAIN)

		# shuffle test data and labels so that batches are not equal every epoch
		# self.log.out("Shuffling TrainData")
		# self.seed   = rnd.randint(len(train_data), size=(len(train_data)//10,2))
		# train_data  = shuffle(train_data, self.seed)
		# if self.classification:
		# 	train_label = shuffle(train_label, self.seed)

		self.log.out("Running Epoch")
		# self.log.info("++ Using Weight Decay! Not updating bias! ++")
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
					sess.run(self.update_l_s[1:-1],{self.temp_tf : self.temp})#self.glauber_step(clamp="v+l",self.temp=self.temp) #sess.run(self.update_l_s[1:-1],{self.temp_tf : self.temp})
				else:
					sess.run(self.update_l_s[1:],{self.temp_tf : self.temp})#self.glauber_step(clamp="v",self.temp=self.temp) #sess.run(self.update_l_s[1:],{self.temp_tf : self.temp})

			# last run calc only the probs to reduce noise
			sess.run(self.update_l_p[1:-1],{self.temp_tf : self.temp})
			# save all layer for bias update
			sess.run(self.assign_save_layer)
			# update the positive gradients
			sess.run(self.update_pos_grad)


			#### Free Running

			for n in range(self.freerun_steps):
				sess.run(self.update_l_s,{self.temp_tf : self.temp})

			# calc probs for noise cancel
			sess.run(self.update_l_p,{self.temp_tf : self.temp})
			# calc he negatie gradients
			sess.run(self.update_neg_grad)




			#### run all parameter updates
			sess.run([self.update_w, self.update_bias], {self.learnrate_tf : self.learnrate})


			#### norm the weights
			if self.DO_NORM_W:
				## simple method
				for i in range(self.n_layers):
					# weights
					if i<(self.n_layers-1) and sess.run(self.len_w[i]) > 1:
						sess.run(self.do_norm_w[i])
					# bias
					if sess.run(self.mean_bias[i]) > 1:
						sess.run(self.do_norm_b[i])


			### calc errors and other things
			ii = self.update//10
			if self.update%10==0 and ii < len(self.updates):
				# self.updates[ii] = self.update
				self.recon_error_train[ii] = (sess.run(self.error,{self.layer_ph[0] : batch}))
				if self.classification:
					self.class_error_train[ii] = (sess.run(self.class_error,{self.layer_ph[-1] : batch_label}))
				self.layer_diversity_train[ii] = (sess.run(self.layer_diversity))
				self.layer_act_train[ii] = (sess.run(self.layer_activities))

				self.l_mean += sess.run(self.layer_activities)

				# check if freerunning escaped fixpoint
				self.freerun_diff_train[ii] = sess.run(self.freerun_diff)


			## update parameters
			self.update += 1
			self.temp = self.get_temp(self.update+self.update_off, self.TEMP_SLOPE, self.TEMP_START, self.TEMP_MIN)

			### liveplot
			if self.liveplot and plt.fignum_exists(fig.number) and start%40==0:
				if start%4000==0:
					ax.cla()
					data = ax.matshow(tile(self.w[0].eval()),vmin=tile(self.w[0].eval()).min()*1.2,vmax=tile(self.w[0].eval()).max()*1.2)

				matrix_new = tile(self.w[0].eval())
				data.set_data(matrix_new)
				plt.pause(0.00001)


		self.log.end() #ending the epoch



		# export the tensorflow vars into numpy arrays
		self.export()


		# calculate diff between all vars
		self.log.out("Calculation weights diff")
		self.weights_diff = []
		self.bias_diff    = []
		for i in range(self.n_layers):
			if i < self.n_layers-1:
				self.weights_diff.append(np.abs(self.w_np[i]-w_old[i]))
			self.bias_diff.append(np.abs(self.bias_np[i]-b_old[i]))


		### write vars into savedict
		self.update_savedict("training", sess)
		self.l_mean[:] = 0

		# increase epoch counter
		self.epochs += 1

		# change self.learnrate
		self.log.info("Learnrate: ",np.round(self.learnrate,6))
		self.learnrate = self.get_learnrate(self.epochs, self.LEARNRATE_SLOPE, self.LEARNRATE_START)

		# print self.temp
		self.log.info("Temp: ",np.round(self.temp,5))
			# self.temp change is inside batch loop

		# change freerun_steps
		self.log.info("freerun_steps: ",self.freerun_steps)
		self.freerun_steps = self.get_N(self.epochs)

		# average layer activities over epochs
		self.l_mean *= 1.0/num_batches



		self.log.reset()

	def test(self, test_data, test_label = None, N = 50):
		""" wrapper function for the old _test_() function"""
		self.test_data = test_data
		self.test_label = test_label

		if self.DO_TESTING:
			with tf.Session() as sess:
				self._test_(
					test_data,
					test_label,
					N               = N,  # sample ist aus random werten, also mindestens 2 sample machen
					create_conf_mat = 0,
					temp_start      = self.temp,
					temp_end        = self.temp,
					sess = sess
					)

	def _test_(self, my_test_data, my_test_label, N, create_conf_mat, temp_start, temp_end, sess):
		"""
		(old test function without session definition, use new wrapper test() instead)
		testing runs without giving h2 , only v is given and h2 has to be infered
		by the DBM
		array my_test_data :: images to test, get assigned to v layer
		int N :: Number of updates from hidden layers

		"""

		### init the vars and reset the weights and biases
		self.batchsize       = len(my_test_data)
		self.learnrate       = self.LEARNRATE_START

		layer_save_test = [[None] for i in range(self.n_layers)]   # save layers while N hidden updates
		for layer in range(len(layer_save_test)):
			layer_save_test[layer] = np.zeros([N, self.batchsize, self.SHAPE[layer]])
		self.layer_act_test  = np.zeros([N,self.n_layers])  # layer activities while N hidden updates

		#temp decrease
		mytemp                 = temp_start
		temp_delta             = (temp_end-temp_start)/float(N)

		droprate = 1000 # 1000 => basically no dropout

		### init the graph
		if self.DO_LOAD_FROM_FILE and not self.DO_TRAINING:
			self.load_from_file(self.workdir+"/data/"+self.PATHSUFFIX,override_params=1)
		self.graph_init("testing", sess) # "testing" because this graph creates the testing variables where only v is given, not h2
		self.import_(sess)


		#### start test run
		self.log.start("Testing DBM with %i images"%self.batchsize)

		# give input to v layer
		sess.run(self.assign_l[0], {self.layer_ph[0] : my_test_data, self.temp_tf : mytemp})

		# update hidden and label N times
		self.log.start("Sampling hidden %i times "%N)
		self.log.info("temp: %f -> %f"%(np.round(temp_start,5),np.round(temp_end,5)))
		# make N clamped updates
		for n in range(N):
			self.layer_act_test[n,:] = sess.run(self.layer_activities, {self.temp_tf : mytemp})
			self.glauber_step("visible", mytemp, droprate, layer_save_test, n, sess) # sess.run(self.update_l_s[1:], {self.mytemp_tf : mytemp})
			# increment mytemp
			mytemp+=temp_delta


		# calc layer variance across batch
		self.layer_diversity_test = sess.run(self.layer_diversity)
		self.layer_variance_test  = sess.run(self.layer_variance)
		# self.unit_diversity_test  = sess.run(self.unit_diversity) # histogram sollte ja um 0.3 verteilt sein, sieht aber nicht besonders aus


		self.log.end()

		## get firerates of every unit and also update all units states to their probabilites
		self.firerate_test = sess.run(self.update_l_p, {self.temp_tf : mytemp, self.droprate_tf : droprate})


		# were firerates are around 0.1
		# self.neuron_good_test_firerate_ind=[None]*DBM.n_layers
		# for l in range(1,DBM.n_layers-1):
		# 	self.neuron_good_test_firerate_ind[l] = np.where((np.mean(self.firerate_test[l],0)>0.02) & (np.mean(self.firerate_test[l],0)<0.4))[0]


		# ### layer input measure from each adjacent layer
		self.l_input_test, self.l_var_test = self.get_total_layer_input(sess)


		### unit input histogram measure
		self.hist_input_test =  self.get_units_input(sess)


		#### reconstruction of v
		# update v M times
		self.label_l_save = layer_save_test[-1][-1][:,:]

		for i in range(N):
			layer_save_test[0][i] = sess.run(self.update_l_s[0],{self.temp_tf : mytemp, self.droprate_tf: droprate})




		#### calculate errors and activations
		self.recon_error  = self.error.eval({self.layer_ph[0] : my_test_data})


		#### count how many images got classified wrong
		self.log.out("Taking only the maximum")
		n_wrongs             = 0
		# label_copy         = np.copy(self.label_l_save)
		wrong_classified_ind = []
		wrong_maxis          = []
		right_maxis          = []


		if self.classification:
			## error of classifivation labels
			self.class_error_test = np.mean(np.abs(self.label_l_save-my_test_label[:,:10]))

			# for i in range(len(self.label_l_save)):
			# 	digit   = np.where(my_test_label[i]==1)[0][0]
			# 	maxi    = self.label_l_save[i].max()
			# 	max_pos = np.where(self.label_l_save[i] == maxi)[0][0]
			# 	if max_pos != digit:
			# 		wrong_classified_ind.append(i)
			# 		wrong_maxis.append(maxi)#
			# 	elif max_pos == digit:
			# 		right_maxis.append(maxi)
			# n_wrongs = len(wrong_maxis)

			if create_conf_mat:
				self.log.out("Making Confusion Matrix")

				self.conf_data = np.zeros([10,1,10]).tolist()

				for i in range(self.batchsize):
					digit = np.where( test_label[i] == 1 )[0][0]

					self.conf_data[digit].append( self.label_l_save[i].tolist() )

				# confusion matrix
				w = np.zeros([10,10])
				for digit in range(10):
					w[digit]  = np.round(np.mean(np.array(DBM.conf_data[digit]),axis=0),3)
				seaborn.heatmap(w*100,annot=True)
				plt.ylabel("Desired Label")
				plt.xlabel("Predicted Label")

		# self.class_error_test = float(n_wrongs)/self.batchsize



		# append test results to save_dict
		self.update_savedict("testing", sess)


		self.tested = 1 # this tells the train function that the batchsize has changed

		self.log.end()
		self.log.info("------------- Test self.log -------------")
		self.log.info("Reconstr. error normal: ",np.round(self.recon_error,5))
		# if self.n_layers==2: self.log.info("Reconstr. error reverse: ",np.round(self.recon_error_reverse,5))
		if self.classification:
			self.log.info("Class error: ",np.round(self.class_error_test, 5))
			self.log.info("Wrong Digits: ",n_wrongs," with average: ",round(np.mean(wrong_maxis),3))
			self.log.info("Correct Digits: ",len(right_maxis)," with average: ",round(np.mean(right_maxis),3))
		self.log.reset()
		return wrong_classified_ind

	def _gibbs_sampling_(self, v_input, gibbs_steps, TEMP_START, temp_end, droprate_start, droprate_end, subspace, mode, liveplot=1, l_input=None):
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
			self.log.info("Liveplotting gibbs sampling")
			fig,ax = plt.subplots(1,self.n_layers+1,figsize=(15,6))
			# plt.tight_layout()

		self.log.start("Gibbs Sampling")
		self.log.info("Mode: %s | Steps: %i"%(mode,gibbs_steps))
		self.log.info("Temp_range:",round(TEMP_START,5),"->",round(temp_end,5))
		self.log.info("Dropout_range:",round(droprate_start,5),"->",round(droprate_end,5))

		if mode == "context":
			sess.run(self.assign_l[0],{self.layer_ph[0] : v_input})
			for i in range(1,self.n_layers):
				sess.run( self.assign_l[i], {self.layer_ph[i] : 0.01*rnd.random([self.batchsize, self.SHAPE[i]])} )


			input_label = test_label[index_for_number_gibbs[:]]


			# set the weights to 0 if context is enebaled and subspace is not "all"
			if subspace == "all":
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


				self.log.out("Setting Weights to 0")
				# get the weights as numpy arrays
				w_ = self.w[-1].eval()
				b_ = self.bias[-1].eval()
				# set values to 0
				# w_[:,subspace_anti] = 0
				b_[subspace_anti] = -1e10
				# assign to tf variables
				# sess.run(self.w[-1].assign(w_))
				sess.run(self.bias[-1].assign(b_))



			### gibbs steps
			for step in range(gibbs_steps):



				if subspace == "all":
					## without context
					#	save layer activites
					self.activity_nc[:,step]  = sess.run(self.layer_activities[1:], {self.temp_tf : temp})
					self.class_error_gibbs_nc[step] = np.mean(np.abs(layer_gs[-1][step-1] - input_label))
					# save layer difference to previous layers
					if step != 0:
						for l in range(1,self.n_layers):
							self.layer_diff_gibbs_nc[l-1, step] = np.mean(np.abs(layer_gs[l][step-1] - layer_gs[l][step]))
				else:
					## wih context
					# save layer activites
					self.activity_c[:,step]   = sess.run(self.layer_activities[1:], {self.temp_tf : temp})
					self.class_error_gibbs_c[step] = np.mean(np.abs(layer_gs[-1][step-1]-input_label))
					# save layer difference to previous layers
					if step!=0:
						for l in range(1,self.n_layers):
							self.layer_diff_gibbs_c[l-1, step] = np.mean(np.abs(layer_gs[l][step-1] - layer_gs[l][step]))


				# update all self.layer except first one (set step = 0 because time series is not saved)
				self.glauber_step("visible", temp, droprate, layer_gs, step) #sess.run(self.update_l_s[1:], {self.temp_tf : temp})

				# assign new temp and dropout rate
				temp += temp_delta
				droprate += drop_delta




			## gather input data # calc layer variance across batch
			if subspace=="all":
				## calc layer probs and set these to the layer vars to smooth later calcs
				self.firerate_nc               = sess.run(self.update_l_p[:],{self.temp_tf : temp, self.droprate_tf : droprate})
				self.l_input_nc, self.l_var_nc = self.get_total_layer_input()
				self.hist_input_nc             = self.get_units_input()
				self.unit_diversity_nc         = sess.run(self.unit_diversity)
				self.layer_diversity_nc        = sess.run(self.layer_diversity)

				# save to file	
				# save_firerates_to_file(self.firerate_nc,saveto_path+"/FireratesNoContext")

			else:
				self.firerate_c              = sess.run(self.update_l_p[:],{self.temp_tf : temp, self.droprate_tf : droprate})
				self.l_input_c, self.l_var_c = self.get_total_layer_input()
				self.hist_input_c            = self.get_units_input()
				self.unit_diversity_c        = sess.run(self.unit_diversity)
				self.layer_diversity_c       = sess.run(self.layer_diversity)

				## save to file
				# save_firerates_to_file(self.firerate_c,saveto_path+"/FireratesContext")

		if mode=="generate":
			sess.run(self.assign_l_zeros)
			sess.run(self.layer[-1].assign(v_input))


			## init save arrays for every layer and every gibbs step
			self.layer_save_generate = [[None] for i in range(self.n_layers)]
			for layer in range(len(self.layer_save_generate)):
				self.layer_save_generate[layer] = np.zeros( [gibbs_steps, self.batchsize, self.SHAPE[layer]] )
			self.temp_save = np.zeros([gibbs_steps,self.batchsize])
			self.energy_generate = np.zeros([gibbs_steps,self.batchsize])

			for step in range(gibbs_steps):
				# update all layer except the last one (labels)
				self.glauber_step("label",temp, droprate, self.layer_save_generate, step) #sess.run(self.update_l_s[:-1], {self.temp_tf : temp})
				self.energy_generate[step] = sess.run(self.energy)
				self.layer_save_generate[-1][step] = v_input

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
				temp     += temp_delta
				droprate += drop_delta

				# save the temp for later Plotting
				self.temp_save[step,:] = temp

			self.unit_input_generate = self.get_units_input();

		if mode=="freerunning":
			sess.run(self.assign_l_zeros)
			if LOAD_MNIST:
				rng  =  index_for_number_test_clear[7][2]#rnd.randint(100)

			# self.log.out("PreAssigning v1 with test data. rng: "+str(rng))
			# sess.run(self.assign_l[0], {self.layer_ph[0] : np.repeat(test_data[rng:rng+1],self.batchsize,0)})
			## run to equlibrium
			for i in range(10):
				sess.run(self.update_l_s[1:],{self.temp_tf : temp, self.droprate_tf : droprate_start})

			## init save arrays for every layer and every gibbs step
			self.layer_save_generate = [[None] for i in range(self.n_layers)]
			for i in range(len(self.layer_save_generate)):
				self.layer_save_generate[i] = np.zeros([gibbs_steps,self.batchsize, DBM.SHAPE[i]])
			self.energy_generate = np.zeros([gibbs_steps,self.batchsize])

			for step in range(gibbs_steps):

				# update all layer
				self.glauber_step("None", temp, droprate, self.layer_save_generate, step) #sess.run(self.update_l_s, {self.temp_tf : temp})
				self.energy_generate[step] = sess.run(self.energy)

				if step == 0 or step == 1 or step == 5 or step == 10 or step == 100 or step == 1000:
					plt.matshow(self.layer_save_generate[0][step][0].reshape(28,28))
					plt.title(str(self.energy_generate[step][0]))


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
				droprate+=drop_delta

		if mode=="clamped":
			sess.run(self.assign_l_zeros)

			self.layer_save_generate = [[None] for i in range(self.n_layers)]
			for layer in range(len(self.layer_save_generate)):
				self.layer_save_generate[layer] = np.zeros( [gibbs_steps, self.batchsize, self.SHAPE[layer]] )
			
			self.energy_generate = np.zeros([gibbs_steps,self.batchsize])

			sess.run(self.layer[0].assign(v_input))
			sess.run(self.layer[-self.n_label_layer].assign(l_input))

			for step in range(gibbs_steps):
				self.glauber_step("v+l", temp, droprate, self.layer_save_generate, step)
				self.energy_generate[step] = sess.run(self.energy)

				# assign new temp
				temp += temp_delta
				droprate+=drop_delta


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

		self.log.end()
		if mode == "context":
			# return the mean of the last 30 gibbs samples for all images
			return np.mean(layer_gs[-1][-30:,:],axis=0)
		else:
			# return the last images that got generated
			return np.mean(self.layer_save_generate[0][-10:],0)
			# v_layer = sess.run(self.update_l_p[0], {self.temp_tf : temp})
			# return v_layer

	def export(self):
		# convert weights and biases to numpy arrays
		self.w_np=[]
		for i in range(len(self.w)):
			self.w_np.append(self.w[i].eval())
		self.bias_np = []
		for i in range(len(self.bias)):
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
		self.log.info("Saved Weights and Biases as NumPy Arrays.")

	def backup_params(self):
		if self.DO_TRAINING and DO_SAVE_TO_FILE:
			new_path = saveto_path+"/Backups/Backup_%i-Error_%f"%(self.epochs,np.round(self.class_error_test, 3))
			if not os.path.isdir(new_path):
				os.makedirs(new_path)
			os.chdir(new_path)

			if self.exported!=1:
				self.export()

			# save weights
			for i in range(len(self.w_np)):
				np.savetxt("w%i.txt"%i, self.w_np[i], fmt = "%1.3e")

			##  save bias
			for i in range(self.n_layers):
				np.savetxt("bias%i.txt"%i, self.bias_np[i], fmt = "%1.3e")

			## save save_dict
			try:
				save_df = DataFrame(dict([ (k,Series(v)) for k,v in self.save_dict.iteritems() ]))
			except:
				self.log.out("using dataframe items conversion for python 3.x")
				save_df = DataFrame(dict([ (k,Series(v)) for k,v in self.save_dict.items() ]))
			save_df.to_csv("save_dict.csv")
		os.chdir(self.workdir)

	def write_to_file(self):
		new_path = self.saveto_path
		if not os.path.isdir(self.saveto_path):
			os.makedirs(new_path)
		os.chdir(new_path)

		if self.DO_TRAINING:
			if self.exported!=1:
				self.export()

			# save weights
			for i in range(len(self.w_np)):
				np.savetxt("w%i.txt"%i, self.w_np[i],fmt="%.5e")

			##  save bias
			for i in range(len(self.bias_np)):
				np.savetxt("bias%i.txt"%i, self.bias_np[i],fmt="%.5e")

			## save save_dict
			try:
				save_df = DataFrame(dict([ (k,Series(v)) for k,v in self.save_dict.iteritems() ]))
			except:
				self.log.out("using dataframe items conversion for python 3.x")
				save_df = DataFrame(dict([ (k,Series(v)) for k,v in self.save_dict.items() ]))
			save_df.to_csv("save_dict.csv")


			## save self.log
			self.log_list["Train_Time"] = self.train_time
			self.log_list["Epochs"]     = self.epochs
			self.log_list["Update"]     = self.update
			self.log.info("Saved data and self.log to:", new_path)

		## wrte logfile
		with open("logfile.txt","w") as log_file:
				for key in self.log_list:
					log_file.write(key+","+str(self.log_list[key])+"\n")


		os.chdir(self.workdir)

	def get_hidden_layer_ind(self):
		return range(1,self.n_layers-self.n_label_layer)

	def show_results(self):

		if self.DO_SHOW_PLOTS:
			
			if self.DO_TRAINING:

				# plot w1 as image
				fig = plt.figure(figsize=(9,9))
				map1 = plt.matshow(tile(self.w_np[0]), cmap = "gray", fignum = fig.number)
				plt.colorbar(map1)
				plt.grid(False)
				plt.title("W %i"%0)
				save_fig(self.saveto_path+"/weights_img.pdf", self.DO_SAVE_TO_FILE)

				# plot layer diversity
				try:
					plt.figure("Layer diversity train")
					for i in range(self.n_layers):
						label_str = get_layer_label(self.type(),self.n_layers, i)
						y = smooth(np.array(self.layer_diversity_train)[:,i],10)
						plt.plot(self.updates[:len(y)],y,label=label_str,alpha=0.7)
						plt.legend()
					plt.xlabel("Update Number")
					plt.ylabel("Deviation")
					save_fig(self.saveto_path+"/layer_diversity.png", self.DO_SAVE_TO_FILE)
				except:
					print("Could not plot layer diversity.")
				
				## train errors
				plt.figure("Errors")
				# plt.plot(self.updates,self.recon_error_train,".",label="Recon Error Train",alpha=0.2)
				if self.classification:
					plt.plot(self.updates,self.class_error_train,".",label="Class Error Train",alpha=0.2)
				## test errors
				x = np.array(self.save_dict["Test_Epoch"]) * self.N_BATCHES_TRAIN
				# plt.plot(x,self.save_dict["Recon_Error"],"o--",label="Recon Error Test")
				if self.classification:
					plt.plot(x,self.save_dict["Class_Error"],"^--",label="Class Error Test")
					try:
						plt.plot(x[:-1],self.save_dict["Class_Error_Train_Data"],"x--",label="Class Error Test\nTrain Data")
					except:
						pass
				plt.legend(loc="best")
				plt.xlabel("Update Number")
				plt.ylabel("Mean Square Error")
				save_fig(self.saveto_path+"/errors.png", self.DO_SAVE_TO_FILE)


				# plot all other weights as hists
				n_weights = len(self.w_np)
				fig,ax    = plt.subplots(n_weights,1,figsize=(8,10),sharex="col")
				for i in range(n_weights):
					if n_weights>1:
						seaborn.distplot(self.w_np[i].flatten(),rug=False,bins=60,ax=ax[i],label="After Training")
						ylim = ax[i].get_ylim()
						ax[i].axvline(self.w_np[i].max(),0,0.2, linestyle="-", color="k")
						ax[i].axvline(self.w_np[i].min(),0,0.2, linestyle="-", color="k")
						# ax[i].hist((self.w_np[i]).flatten(),bins=60,alpha=0.5,label="After Training")
						ax[i].set_title("W %i"%i)
						# ax[i].set_ylim(-ylim[1]/5,ylim[1])
						ax[i].legend()
					else:
						ax.hist((self.w_np[i]).flatten(),bins=60,alpha=0.5,label="After Training")
						ax.set_title("W %i"%i)
						ax.legend()

					try:
						seaborn.distplot(self.w_np_old[i].flatten(),rug=False,bins=60,ax=ax[i],label="Before Training",color="r")
						# ax[i].hist((self.w_np_old[i]).flatten(),color="r",bins=60,alpha=0.5,label="Before Training")
					except:
						pass
				plt.tight_layout()
				save_fig(self.saveto_path+"/weights_hist.pdf", self.DO_SAVE_TO_FILE)
				try:
					# plot change in w1
					fig=plt.figure(figsize=(9,9))
					plt.matshow(tile(self.w_np[0]-self.w_np_old[0]),fignum=fig.number)
					plt.colorbar()
					plt.title("Change in W1")
					save_fig(self.saveto_path+"/weights_change.pdf", self.DO_SAVE_TO_FILE)
				except:
					plt.close(fig)

				# plot freerun diffs
				fig,ax = plt.subplots(1,1)
				for i in range(self.n_layers):
					l_str = get_layer_label(self.type(),self.n_layers,i)
					ax.semilogy(self.updates, self.freerun_diff_train[:,i], label = l_str)
				plt.ylabel("self.log("+r"$\Delta L$"+")")
				plt.legend(ncol = 2, loc = "best")
				plt.xlabel("Update")
				save_fig(self.saveto_path+"/freerunn-diffs.png", self.DO_SAVE_TO_FILE)


				# plot train data (temp, diffs, learnrate, ..)
				fig,ax = plt.subplots(4,1,sharex="col",figsize=(8,8))

				ax[0].plot(self.save_dict["Temperature"],label="Temperature")
				ax[0].legend(loc="center left",bbox_to_anchor = (1.0,0.5))

				ax[0].set_ylabel("Temperature")

				ax[1].plot(self.save_dict["Learnrate"],label="Learnrate")
				ax[1].legend(loc="center left",bbox_to_anchor = (1.0,0.5))
				ax[1].set_ylabel("Learnrate")

				ax[2].set_ylabel("Mean")
				for i in range(len(self.w)):
					ax[2].plot(self.save_dict["W_mean_%i"%i],label="W %i"%i)
				ax[2].legend(loc="center left",bbox_to_anchor = (1.0,0.5))

				ax[3].set_ylabel("Diff")
				if self.epochs>2:
					for i in range(len(self.SHAPE)-1):
						ax[3].semilogy(self.save_dict["W_diff_%i"%i][1:],label = "W %i"%i)
					for i in range(len(self.SHAPE)):
						ax[3].semilogy(self.save_dict["Bias_diff_%i"%i][1:],label = "Bias %i"%i)

				ax[3].legend(loc="center left",bbox_to_anchor = (1.0,0.5))

				ax[-1].set_xlabel("Epoch")

				plt.subplots_adjust(bottom=None, right=0.73, left=0.2, top=None,
							wspace=None, hspace=None)

				save_fig(self.saveto_path+"/learnr-temp.pdf", self.DO_SAVE_TO_FILE)




				plt.figure("Layer_activiations_train_run")
				for i in range(self.n_layers):
					label_str = get_layer_label(self.type(),self.n_layers, i)
					plt.plot(self.updates,np.array(self.layer_act_train)[:,i],label = label_str)
				plt.legend()
				plt.xlabel("Update Number")
				plt.ylabel("Train Layer Activity in %")
				save_fig(self.saveto_path+"/layer_act_train.png", self.DO_SAVE_TO_FILE)

			if self.DO_TESTING:
				# plot some samples from the testdata
				fig3,ax3 = plt.subplots(len(self.SHAPE)+1,13,figsize=(16,8),sharey="row")
				for i in range(13):
					# plot the input
					if self.type() != "DBM_attention":
						ax3[0][i].matshow(self.test_data[i:i+1].reshape(int(sqrt(self.SHAPE[0])),int(sqrt(self.SHAPE[0]))))
						ax3[1][i].matshow(self.firerate_test[0][i].reshape(int(sqrt(self.SHAPE[0])),int(sqrt(self.SHAPE[0]))))
					else:
						ax3[0][i].matshow(test_data_attention[i].reshape(28*2,28))
						ax3[0][i].set_title(str(test_label_attention_side[i]))
						ax3[1][i].matshow(self.firerate_test[0][i].reshape(28*2,28))	
				
					ax3[0][i].set_yticks([])
					ax3[0][i].set_xticks([])
				
					ax3[1][i].set_yticks([])
					ax3[1][i].set_xticks([])

					#plot hidden layer
					for layer in self.get_hidden_layer_ind():
						try:
							ax3[layer+1][i].matshow(self.firerate_test[layer][i].reshape(int(sqrt(self.SHAPE[layer])),int(sqrt(self.SHAPE[layer]))))
							ax3[layer+1][i].set_yticks([])
							ax3[layer+1][i].set_xticks([])
						except:
							pass
					# plot the last layer
					if self.classification:
						color = "r"
						
						ax3[-self.n_label_layer][i].bar(range(self.SHAPE[-self.n_label_layer]),self.label_l_save[i],color=color, label= "Predicted")
						ax3[-self.n_label_layer][i].bar(range(self.SHAPE[-self.n_label_layer]),self.test_label[i],color="gray", alpha = 0.5, label= "Desired")

						ax3[-self.n_label_layer][i].set_xticks(range(self.SHAPE[-self.n_label_layer]))
						ax3[-self.n_label_layer][i].set_ylim(0,1)

						
					else:
						ax3[-1][i].matshow(self.label_l_save[i].reshape(int(sqrt(self.SHAPE[-1])),int(sqrt(self.SHAPE[-1]))))
						ax3[-1][i].set_xticks([])
						ax3[-1][i].set_yticks([])

					#plot the reconstructed layer h1
					# ax3[5][i].matshow(self.rec_h1[i:i+1].reshape(int(sqrt(self.SHAPE[1])),int(sqrt(self.SHAPE[1]))))
					# plt.matshow(random_recon.reshape(int(sqrt(self.SHAPE[0])),int(sqrt(self.SHAPE[0]))))
				plt.tight_layout(pad=0.2)
				plt.legend()
				save_fig(self.saveto_path+"/examples.pdf", self.DO_SAVE_TO_FILE)
			
			
			plt.show()




################################################################################################################################################
### Class Extra Unit DBM
class DBM_attention_class(DBM_class):
	""" 
	Defines a DBM with an addition "attention" layer (which is always defined as the last layer). 
	This DBM receives 2 images side by side. 
	This layer acts like a visible label layer, but the labels tell if right oder left v1 is attented.
	"""

	def __init__ (self, shape, liveplot, classification, UserSettings):
		""" 
		Inherits many things from DBM_class.

		* new variables: *
		layers_to_connect :: array or list of the layer index to connect the context layer to 
								- the label layer has to be exluded, only include additional layers
		"""
		# define the parent class
		self.parent = super(DBM_attention_class, self);

		# call init from parent
		self.parent.__init__(shape, liveplot, classification, UserSettings)

		# hich layer to connect the context layer to
		self.layers_to_connect = np.array(self.LAYERS_TO_CONNECT)
		# how many connected layers to context layer
		self.n_context_con     = len(self.layers_to_connect)
		# how many units in context layer
		self.n_context_units   = self.SHAPE[-1]
		# how many label layer the system has
		self.n_label_layer     = 2

		self.save_dict["Context_Error"] = []
		for j in range(len(self.layers_to_connect)):
			i = j+len(self.DBM_SHAPE)-1
			self.save_dict["W_mean_%i"%i] = []
			self.save_dict["W_diff_%i"%i] = []
			self.save_dict["CD_abs_mean_%i"%i] = []

	def type(self):
		return "DBM_attention"

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
		if clamp == "visible + label" or clamp == "v+l":
			rnd_order = list(range(1,self.n_layers-2))
		
		elif clamp == "None":
			rnd_order = list(range(0,self.n_layers))
		
		elif clamp == "visible" or clamp == "v":
			rnd_order = list(range(1,self.n_layers))
		
		elif clamp == "label" or clamp == "l":
			rnd_order = list(range(0,self.n_layers-2))

		elif clamp == "visible + context":
			rnd_order = list(range(1,self.n_layers-1))


		# shuffle the list to really make it random
		rnd.shuffle(rnd_order)

		# run the updates in that random order
		for layer in rnd_order:
			if save_array == "None":
				sess.run(self.update_l_s[layer], {self.temp_tf : temp, self.droprate_tf : droprate})
			else:
				save_array[layer][step] = sess.run(self.update_l_s[layer], {self.temp_tf : temp, self.droprate_tf : droprate})

	def layer_input(self, layer_i):
		""" calculate input of layer layer_i
		layer_i :: for which layer
		returns :: input for the layer - which are the probabilites
		"""
		# v(1) layer
		if layer_i == 0:
			w = self.w[layer_i];
			if self.USE_DROPOUT:
				w *= self.dropout_matrix[layer_i]
			_input_ = sigmoid(tf.matmul(self.layer[layer_i+1], w,transpose_b=True) + self.bias[layer_i], self.temp_tf)

		# v(3) layer 
		elif layer_i == self.n_layers-1:
			### add all termns together to an buffer 
			# first connection between label and context layer 
			w = self.w[layer_i-1]
			if self.USE_DROPOUT:
				w *= self.dropout_matrix[layer_i-1]
			_input_buff_ = tf.matmul(self.layer[layer_i-1], w)
			# every extra layer
			for i,l in enumerate(self.layers_to_connect):
				index = i + self.n_layers-1
				w = self.w[index]
				if self.USE_DROPOUT:
					w *= self.dropout_matrix[index]
				_input_buff_ += tf.matmul(self.layer[l], w)
			# bias 
			_input_buff_ += self.bias[layer_i]

			# sigmoid 
			_input_ = sigmoid(_input_buff_, self.temp_tf)

		# hidden and v(2) layer 
		else:
			# normal layer bottom up and top down adjacend 
			w0 = self.w[layer_i-1];
			w1 = self.w[layer_i];
			if self.USE_DROPOUT:
				w0 *= self.dropout_matrix[layer_i-1]
				w1 *= self.dropout_matrix[layer_i]
			_input_buff_ = (tf.matmul(self.layer[layer_i-1],w0)
								+ tf.matmul(self.layer[layer_i+1],w1,transpose_b=True)
								+ self.bias[layer_i])

			# extra layer input :
			correct_extra_index = self.layers_to_connect+len(DBM.SHAPE)
			if layer_i in correct_extra_index:
				w_index = DBM.n_layers-1 + np.where(correct_extra_index == layer_i)[0][0]
				w2 = self.w[w_index]
				if self.USE_DROPOUT:
					w2 *= self.dropout_matrix[w_index]
				_input_buff_ += tf.matmul(self.layer[-1], w2, transpose_b = True)				

			_input_ = sigmoid(_input_buff_, self.temp_tf)			

		return _input_

	def layer_input_context(self):
		"""
		!NOT USED!
		contruct input for the context layer
		!NOT USED!
		"""
		_input_ = 0
		for l in self.layers_to_connect:
			_input_ = _input_ + tf.matmul(self.layer[l], self.w_context[l])
		_input_ = _input_ + self.bias_context

		_input_ = sigmoid(_input_, self.temp_tf)

		return _input_

	def graph_init(self, graph_mode):
		# parent graph init (also creates new matrices used for contex layer)
		self.parent.graph_init(graph_mode)

		# sample the label layer
		self.update_l_s[-2] = self.layer[-2].assign(self.layer_prob[-2])

		self.context_error_tf = tf.reduce_mean(tf.square(self.layer_ph[-1]-self.layer[-1]))
		self.class_error = tf.reduce_mean(tf.square(self.layer_ph[-2]-self.layer[-2]))

				
		sess.run(tf.global_variables_initializer())

	def train(self, train_data, train_label, train_label_attention, num_batches, cont):
		""" training the DBM with given h2 as labels and v as input images
		train_data  :: images
		train_label :: corresponding label
		num_batches :: how many batches
		"""
		######## init all vars for training
		self.batchsize = int(len(train_data)/num_batches)
		self.num_of_updates = N_EPOCHS_TRAIN*num_batches


		# number of clamped sample steps
		if self.n_layers <=3 and self.classification == 1:
			M = 2
		else:
			M = 20




		### free energy
		# self.F=[]
		# self.F_test=[]

		if self.DO_LOAD_FROM_FILE and not cont:
			# load data from the file
			self.load_from_file(workdir+"/data/"+self.PATHSUFFIX,override_params=1)
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
		for i in range(len(self.w)):
			w_old.append(np.copy(self.w[i].eval()))
		for i in range(len(self.bias)):
			b_old.append(np.copy(self.bias[i].eval()))

		# starting the training
		log.info("Batchsize:",self.batchsize,"N_Updates",self.num_of_updates)

		log.start("Deep BM Epoch:",self.epochs+1,"/",N_EPOCHS_TRAIN)

		# shuffle test data and labels so that batches are not equal every epoch
		log.out("Shuffling TrainData")
		self.seed   = rnd.randint(len(train_data),size=(len(train_data)//10,2))
		train_data  = shuffle(train_data, self.seed)
		if self.classification:
			train_label = shuffle(train_label, self.seed)
			train_label_attention = shuffle(train_label_attention, self.seed)

		log.out("Running Epoch")
		# log.info("++ Using Weight Decay! Not updating bias! ++")
		# self.persistent_layers = sess.run(self.update_l_s,{self.temp_tf : temp})

		for start, end in zip( range(0, len(train_data), self.batchsize), range(self.batchsize, len(train_data), self.batchsize)):
			# define a batch
			batch = train_data[start:end]
			if self.classification:
				batch_label = train_label[start:end]
				batch_label_context = train_label_attention[start:end]




			#### Clamped Run

			# assign v and h2 to the batch data
			sess.run(self.assign_l[0], { self.layer_ph[0]  : batch })
			if self.classification:
				sess.run(self.assign_l[-2], {self.layer_ph[-2] : batch_label})
				sess.run(self.assign_l[-1], {self.layer_ph[-1] : batch_label_context})

			# calc hidden layer samples (not the visible & label layer)
			for hidden in range(M):
				if self.classification:
					sess.run(self.update_l_s[1:-2],{self.temp_tf : self.temp})#self.glauber_step(clamp="v+l",self.temp=self.temp) #sess.run(self.update_l_s[1:-1],{self.temp_tf : self.temp})
				else:
					sess.run(self.update_l_s[1:],{self.temp_tf : self.temp})#self.glauber_step(clamp="v",self.temp=self.temp) #sess.run(self.update_l_s[1:],{self.temp_tf : self.temp})

			# last run calc only the probs to reduce noise
			sess.run(self.update_l_p[1:-2],{self.temp_tf : self.temp})
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

			for n in range(self.freerun_steps):
				sess.run(self.update_l_s,{self.temp_tf : self.temp})#self.glauber_step(clamp = "None",self.temp=self.temp) #sess.run(self.update_l_s,{self.temp_tf : self.temp})



			# calc probs for noise cancel
			sess.run(self.update_l_p,{self.temp_tf : self.temp})
			# sess.run(self.update_particles)


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

				## other method (does not work)
				# sess.run(self.do_norm_w)
				# sess.run(self.do_norm_b)


			### calc errors and other things
			ii = self.update//10
			if self.update%10==0 and ii < len(self.updates):
				# self.updates[ii] = self.update
				self.recon_error_train[ii] = (sess.run(self.error,{self.layer_ph[0] : batch}))
				if self.classification:
					self.class_error_train[ii] = (sess.run(self.class_error,{self.layer_ph[-2] : batch_label}))
				self.layer_diversity_train[ii] = (sess.run(self.layer_diversity))
				self.layer_act_train[ii] = (sess.run(self.layer_activities))

				self.l_mean += sess.run(self.layer_activities)

				# check if freerunning escaped fixpoint
				self.freerun_diff_train[ii] = sess.run(self.freerun_diff)


			## update parameters
			self.update += 1

			self.temp = self.get_temp(self.update+self.update_off, TEMP_SLOPE, TEMP_START, TEMP_MIN)

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
		for i in range(len(self.w_np)):
			self.weights_diff.append(np.abs(self.w_np[i]-w_old[i]))
		for i in range(len(self.bias_np)):
			self.bias_diff.append(np.abs(self.bias_np[i]-b_old[i]))


		### write vars into savedict
		self.update_savedict("training")
		self.l_mean[:] = 0

		# increase epoch counter
		self.epochs += 1

		# change self.learnrate
		log.info("Learnrate: ",np.round(self.learnrate,6))
		self.learnrate = self.get_learnrate(self.epochs, self.LEARNRATE_SLOPE, self.LEARNRATE_START)

		# print self.temp
		log.info("Temp: ",np.round(self.temp,5))
			# self.temp change is inside batch loop

		# change freerun_steps
		log.info("freerun_steps: ",self.freerun_steps)
		self.freerun_steps = self.get_N(self.epochs)

		# average layer activities over epochs
		self.l_mean *= 1.0/num_batches



		log.reset()

	def test(self, my_test_data, my_test_label, N, create_conf_mat, temp_start, temp_end, using_train_data = False):
		"""
		testing runs without giving h2 , only v is given and h2 has to be infered
		by the DBM
		array my_test_data :: images to test, get assigned to v layer
		int N :: Number of updates from hidden layers

		"""

		### init the vars and reset the weights and biases
		self.batchsize       = len(my_test_data)
		self.learnrate       = self.get_learnrate(self.epochs, self.LEARNRATE_SLOPE, self.LEARNRATE_START)

		layer_save_test = [[None] for i in range(self.n_layers)]   # save layers while N hidden updates
		for layer in range(len(layer_save_test)):
			layer_save_test[layer] = np.zeros([N, self.batchsize, self.SHAPE[layer]])
		self.layer_act_test  = np.zeros([N,self.n_layers])  # layer activities while N hidden updates

		#temp decrease
		mytemp                 = temp_start
		temp_delta             = (temp_end-temp_start)/float(N)

		droprate = 100 # 100 => basically no dropout

		### init the graph
		if self.DO_LOAD_FROM_FILE and not self.DO_TRAINING:
			self.load_from_file(workdir+"/data/"+self.PATHSUFFIX,override_params=1)
		self.graph_init("testing") # "testing" because this graph creates the testing variables where only v is given, not h2
		self.import_()


		#### start test run
		log.start("Testing DBM with %i images"%self.batchsize)
		if using_train_data:
			log.info("Using train data")
		else:
			log.info("Using test data")

		# give input to v layer
		sess.run(self.assign_l[0], {self.layer_ph[0] : my_test_data})
		sess.run(self.assign_l[-1], {self.layer_ph[-1] : test_label_attention_side})

		# update hidden and label N times
		log.start("Sampling hidden %i times "%N)
		log.info("temp: %f -> %f"%(np.round(temp_start,5),np.round(temp_end,5)))
		
		# make N clamped updates
		for n in range(N):
			# get layer activities 
			self.layer_act_test[n,:] = sess.run(self.layer_activities, {self.temp_tf : mytemp})

			# update layer 
			self.glauber_step("visible + context", mytemp, droprate, layer_save_test, n) # sess.run(self.update_l_s[1:], {self.mytemp_tf : mytemp})
			
			# increment mytemp
			mytemp+=temp_delta



		# calc layer variance across batch
		self.layer_diversity_test = sess.run(self.layer_diversity)
		self.layer_variance_test  = sess.run(self.layer_variance)
		# self.unit_diversity_test  = sess.run(self.unit_diversity) # histogram sollte ja um 0.3 verteilt sein, sieht aber nicht besonders aus


		log.end()

		## get firerates of every unit
		self.firerate_test = sess.run(self.update_l_p, {self.temp_tf : mytemp, self.droprate_tf : droprate})

		# were firerates are around 0.1
		# self.neuron_good_test_firerate_ind=[None]*DBM.n_layers
		# for l in range(1,DBM.n_layers-1):
		# 	self.neuron_good_test_firerate_ind[l] = np.where((np.mean(self.firerate_test[l],0)>0.02) & (np.mean(self.firerate_test[l],0)<0.4))[0]


		# ### layer input measure from each adjacent layer
		self.l_input_test, self.l_var_test = self.get_total_layer_input()


		### unit input histogram measure
		self.hist_input_test =  self.get_units_input()


		#### reconstruction of v
		# update v M times
		self.label_l_save = layer_save_test[-2][-1]
		# self.context_l_save = layer_save_test[-1][-1]

		for i in range(N):
			layer_save_test[0][i] = sess.run(self.update_l_s[0],{self.temp_tf : mytemp, self.droprate_tf: droprate})




		#### calculate errors and activations
		self.recon_error  = self.error.eval({self.layer_ph[0] : my_test_data})


		#### count how many images got classified wrong
		log.out("Taking only the maximum")
		n_wrongs             = 0
		# label_copy         = np.copy(self.label_l_save)
		wrong_classified_ind = []
		wrong_maxis          = []
		right_maxis          = []


		if self.classification:
			## error of classifivation labels
			self.class_error_test = np.mean(np.abs(self.label_l_save - my_test_label))
			self.context_error_test = np.mean(np.abs(self.firerate_test[-1] - test_label_attention_side))


			for i in range(len(self.label_l_save)):
				digit   = np.where(my_test_label[i]==1)[0][0]
				maxi    = self.label_l_save[i].max()
				max_pos = np.where(self.label_l_save[i] == maxi)[0][0]
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

					self.conf_data[digit].append( self.label_l_save[i].tolist() )

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
			log.info("Context Error: ", np.round(self.context_error_test,5))
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
				:: "context" use context units clamped to induce context

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
			if subspace != "all":
				sess.run(self.assign_l[-1], {self.layer_ph[-1] : test_label_context[index_for_number_gibbs[:]]})

			for i in range(1,self.n_layers):
				sess.run( self.assign_l[i], {self.layer_ph[i] : 0.01*rnd.random([self.batchsize, self.SHAPE[i]])} )


			input_label = test_label[index_for_number_gibbs[:]]


			# set the weights to 0 if context is enebaled and if DBM type and subspace is not "all"
			if subspace == "all":
				self.activity_nc  = np.zeros([self.n_layers-1,gibbs_steps]);
				self.layer_diff_gibbs_nc = np.zeros([self.n_layers-1,gibbs_steps])
				self.class_error_gibbs_nc = np.zeros([gibbs_steps]);

			else:
				self.activity_c   = np.zeros([self.n_layers-1,gibbs_steps]);
				self.layer_diff_gibbs_c = np.zeros([self.n_layers-1,gibbs_steps])
				self.class_error_gibbs_c = np.zeros([gibbs_steps]);
				# get all numbers that are not in subspace
				# subspace_anti = []
				# for i in range(10):
				# 	if i not in subspace:
				# 		subspace_anti.append(i)




			### gibbs steps
			for step in range(gibbs_steps):

				if subspace == "all":
					## without context

					# step without clamped context layer 
					self.glauber_step("visible", temp, droprate, layer_gs, step) #sess.run(self.update_l_s[1:], {self.temp_tf : temp})

					#	save layer activites
					self.activity_nc[:,step]  = sess.run(self.layer_activities[1:], {self.temp_tf : temp})
					self.class_error_gibbs_nc[step] = np.mean(np.abs(layer_gs[-2][step]-input_label))

					# save layer difference to previous layers
					if step != 0:
						for i in range(1,self.n_layers):
							self.layer_diff_gibbs_nc[i-1, step] = np.mean(np.abs(layer_gs[i][step-1] - layer_gs[i][step]))
				else:
					## wih context

					# step with clamped context layer 
					self.glauber_step("visible + context", temp, droprate, layer_gs, step) #sess.run(self.update_l_s[1:], {self.temp_tf : temp})

					# save layer activites
					self.activity_c[:,step]   = sess.run(self.layer_activities[1:], {self.temp_tf : temp})
					self.class_error_gibbs_c[step] = np.mean(np.abs(layer_gs[-2][step]-input_label))
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
				self.firerate_nc               = sess.run(self.update_l_p[:],{self.temp_tf : temp, self.droprate_tf : droprate})
				self.l_input_nc, self.l_var_nc = self.get_total_layer_input()
				self.hist_input_nc             = self.get_units_input()
				self.unit_diversity_nc         = sess.run(self.unit_diversity)
				self.layer_diversity_nc        = sess.run(self.layer_diversity)
			else:
				self.firerate_c              = sess.run(self.update_l_p[:],{self.temp_tf : temp, self.droprate_tf : droprate})
				self.l_input_c, self.l_var_c = self.get_total_layer_input()
				self.hist_input_c            = self.get_units_input()
				self.unit_diversity_c        = sess.run(self.unit_diversity)
				self.layer_diversity_c       = sess.run(self.layer_diversity)


		if mode=="generate":
			sess.run(self.assign_l_zeros)
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
			sess.run(self.assign_l_zeros)
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
			return np.mean(layer_gs[-2][-30:,:],axis=0)


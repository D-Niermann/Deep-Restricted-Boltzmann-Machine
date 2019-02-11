# load data 
if "train_data" not in globals():
	if LOAD_MNIST:
		log.out("Loading Data")
		mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
		train_data, train_label, test_data, test_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
		### multiply the label vectors 
		label_mult  = 1
		test_label  = np.concatenate((test_label,)*label_mult,axis=1)
		train_label = np.concatenate((train_label,)*label_mult,axis=1)

		

		# get test data of only one number class:
		index_for_number_test  = np.zeros([10,1200])
		where = np.zeros(10).astype(np.int)
		for i in range(len(test_label)):
			for digit in range(10):
				d_array = np.zeros(10)
				d_array[digit] = 1
				if (test_label[i]==d_array).sum()==10:
					index_for_number_test[digit][where[digit]]=i
					where[digit]+=1	
					break	
		# convert to integer 
		index_for_number_test = index_for_number_test.astype(np.int)
		# clear up the zeros at the end of the arrays
		index_for_number_test_clear = [None]*10
		for digit in range(10):
			index_for_number_test_clear[digit] = np.delete(index_for_number_test[digit],np.where(index_for_number_test[digit]==0)[0][1:])
		
		# delete the dirty data
		del index_for_number_test

		### create MNIST data for the attention task
		# search the train data for 0s, 1s and 2s 
		index_for_number_train  = np.zeros([3,6200])
		where = np.zeros(3).astype(np.int)
		for i in range(len(train_label)):
			for digit in range(3):
				d_array = np.zeros(10)
				d_array[digit] = 1
				if (train_label[i]==d_array).sum()==10:
					index_for_number_train[digit][where[digit]]=i
					where[digit]+=1	
					break	
		# convert to integer
		index_for_number_train = index_for_number_train.astype(np.int)
		# clear up the zeros at the end of the arrays
		index_for_number_train_clear = [None]*3
		for digit in range(3):
			index_for_number_train_clear[digit] = np.delete(index_for_number_train[digit],np.where(index_for_number_train[digit]==0)[0][1:])
		del index_for_number_train

		## combine images from MNIST
		# how many datapoints to create (contains labels and images and train + test data)
		n_data_points               = 50000
		# data with two images side by side
		train_data_attention        = np.zeros([n_data_points,28*28*2])
		# class label of the focused image
		train_label_attention_class = np.zeros([n_data_points,3])
		# label which tells on which image to focus 
		train_label_attention_side  = np.zeros([n_data_points,2])

		for i in range(n_data_points):
			# random class and image and side index 
			c   = rnd.randint(0,3)
			c2  = rnd.randint(0,3)
			while c2 == c:
				c2  = rnd.randint(0,3)
			im  = rnd.randint(0, len(index_for_number_train_clear[c]))
			im2 = rnd.randint(0, len(index_for_number_train_clear[c2]))
			s = rnd.randint(0,2)
			# set the label 
			train_label_attention_side[i][s] = 1 
			if s == 0:
				train_label_attention_class[i][c] = 1	# left side attendet (or top side)
			else:
				train_label_attention_class[i][c2] = 1	# right side attendet
			# combine both images and add them to the dataset
			train_data_attention[i] = np.concatenate([train_data[index_for_number_train_clear[c][im]], train_data[index_for_number_train_clear[c2][im2]]])

		## split off the test dataset
		n_test_points = 10000
		# test label
		test_label_attention_side  = train_label_attention_side[(n_data_points - n_test_points):]
		test_label_attention_class = train_label_attention_class[(n_data_points - n_test_points):]
		# test data
		test_data_attention = train_data_attention[(n_data_points - n_test_points):]
		# for i in range(len(test_data_attention)):
		# 	c2  = rnd.randint(0,3)
		# 	im  = rnd.randint(0, len(index_for_number_train_clear[c2]))
		# 	if test_label_attention_side[i][0] == 1:
		# 		test_data_attention[i][784:] = train_data[index_for_number_train_clear[c2][im]]
		# 	else:
		# 		test_data_attention[i][:784] = train_data[index_for_number_train_clear[c2][im]]

		# delete the test data from the train data arrays
		train_data_attention        = np.delete(train_data_attention, range((n_data_points - n_test_points), n_data_points), axis=0)
		train_label_attention_class = np.delete(train_label_attention_class, range((n_data_points - n_test_points), n_data_points), axis=0)
		train_label_attention_side  = np.delete(train_label_attention_side, range((n_data_points - n_test_points), n_data_points), axis=0)

	

		test_data_noise = sample_np(test_data + (rnd.random(test_data.shape)-0.5)*1.1)

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

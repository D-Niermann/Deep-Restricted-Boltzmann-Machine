# Deep-Restricted-Boltzmann-Machine
Python Codes for my Master Thesis about Deep Restricted Boltzmann Machines


## Structure and Usage
### Files
the main.py file needs to be executed. In there, a DBM is defined. After loading the train data and user settings, the actual code for the DBM is very easy:



```python
#### Create a DBM  #####

DBM = DBM_class( UserSettings = UserSettings,
				logger = log,
				workdir = workdir,
				saveto_path = saveto_path,
				liveplot = 0
)


# pretrain each RBM seperately 
DBM.pretrain(train_data)

# train the complete DBM with train data (test data is used to 
## test the train process between every n'th epochs)
DBM.train(train_data, test_data, train_label, test_label)

# test the DBM with test data
DBM.test(test_data, test_label, N = 50)

# Plot the results or save them to file (see Settings.py)
DBM.show_results()
```

Instead of the DBM class, the RBM class, Shape_BM or the DBM_attention class is also available although not bug free and was only programmed for prototyping and research.
<asd>
### User Settings
These Settings can be used in a default DBM
```python
""" These settings get loaded in by a DBM instance and all variables are copied into that. """

UserSettings = {
	
	#### Which Dataset to load 
		"DO_CLASSIFICATION" : 1,			# if the last layer is a classification layer or not
	
	# training process
		"N_BATCHES_PRETRAIN" : 200, 			# how many batches per epoch for pretraining
		"N_BATCHES_TRAIN"    : 200, 			# how many batches per epoch for complete DBM training
		"N_EPOCHS_PRETRAIN"  : [0,0,0,0,0,0], 		# pretrain epochs for each RBM
		"N_EPOCHS_TRAIN"     : 200, 			# how often to iter through the test images
		"TEST_EVERY_EPOCH"   : 50, 			# how many epochs to train before testing on the test data

	### learnrates
		"LEARNRATE_PRETRAIN" : 0.002,			# learnrate for pretraining
		"LEARNRATE_START"    : 0.003,			# starting learnrate
		"LEARNRATE_SLOPE"    : 1 ,			# bigger number -> more constant learnrate

	# freerunning steps 
		# (replaced in get_N() after first run)
		"N_FREERUN_START" : 2,				# how many freerunning steps to make (gets reset after 1 epoch in get_N() )

	### temperature
		"TEMP_START"    : 1,				# starting temp
		"TEMP_SLOPE"    : 0 , 				# linear decrease slope higher number -> fast cooling
		"TEMP_MIN"      : 1,				# how low temp can fall at minimum

	# seed for random number generation,
	# set None to have no seed 
		"SEED" : None,					# random seed for tf and np

	### state vars
		"DO_PRETRAINING" : 1,				# if no pretrain then files are automatically loaded
		"DO_TRAINING"    : 1,				# if to train the whole DBM
		"DO_TESTING"     : 1,				# if testing the DBM with test data
		"DO_SHOW_PLOTS"  : 1,				# if plots will show on display - either way they get saved into saveto_path

		## used only in DBM.gibbs_sampling() 
			"DO_CONTEXT"    : 0,				# if to test the context 
				"SUBSPACE"  : [0, 1, 2, 3, 4], 		# global subspace set 
			"DO_GEN_IMAGES" : 0,				# if to generate images (mode can be choosen at function call)
				"FREERUN_MODE" : "freerunning",		# Mode of the gibbs sampler to generate images (clamped, freerunning, generate, context)
			"DO_NOISE_STAB" : 0,				# if to make a noise stability test,

		"USE_DROPOUT"  : 1,					# if to use synnaptic failure while training
		"DROPOUT_RATE" : 1,					# multiplication of random uniform synaptic failure matrix (higher number -> less failure)

		"DO_NORM_W"    : 0,					# if to norm the weights and biases to 1 while training

	### saving and loading
		"DO_SAVE_TO_FILE"       : 1, 		# if to save plots and data to file
		"DO_SAVE_PRETRAINED"    : 0, 		# if to save the pretrained weights seperately (for later use)
		"DO_LOAD_FROM_FILE"     : 0, 		# if to load weights and biases from datadir + pathsuffix
		"PATHSUFFIX"            : "Mon_Jun__4_15-55-25_2018_[784, 225, 225, 225, 10] - ['original'] 15%", 
		"PATHSUFFIX_PRETRAINED" : "Thu_Jun__7_13-49-25_2018",


		"DBM_SHAPE" : [	3*3,
						2],

	## only used in DBM_context class
		"LAYERS_TO_CONNECT" : [],		# layer index of which layer to connect the context layer to (v2 label layer is always connected)
}
```
## Results
### Generation
Boltzmann Machines learn to recreate learned data in a stochastic way, so that also data can be created that it has never seen before. Since there are top-down connections in every BM, I could use these to govern which images should be recreated. Starting from a random initial state of every neuron and clamping one class neuron in the last layer to 1 while leaving every other class neuron to 0, i effectivley forced the network to drive into states of that one class. Below is an image where every class was clamped 10 times each. Each row represents the clamped class and each column the trials.
![alt text](https://raw.githubusercontent.com/D-Niermann/Deep-Restricted-Boltzmann-Machine/master/Results/Generierung%20mit%20ohne%20dropout/Thu_Jul_12_12-19-26_2018_%5B784%2C%20225%2C%20225%2C%20225%2C%2010%5D%20-%20%5B'doppelte%20recurents'%5D%20auch%20gut/generated_img.png "Generated images from a deep BM.")

### Discrimination
A 3 layers DBM (1 hidden layer) could get classification errors of around 6% without any optimazation like backpropagation or rolling ball methods. Only the "normal" BM learning rule proposed by G.Hinton, called Contrastive Divergence was used. There are other puplications where this error is even smaller.

### Neuron clustering
The weights between the first and second layer are (biological) plausible and are sometimes used to initialize a perzeptron or other neural networks. Under the following link is a image of the learned weights of one DBM.
<embed>https://github.com/D-Niermann/Deep-Restricted-Boltzmann-Machine/blob/master/DBM%20Project/data/Mon_Jun__4_15-55-25_2018_%5B784%2C%20225%2C%20225%2C%20225%2C%2010%5D%20-%20%5B'original'%5D%2015%25/weights_img.pdf</embed>

Also, extracting information about single neurons reveals that they learned to cluster classes the more the deeper the layers are (which makes a lot of sense but still). 
Here is and typical neuron of the third hidden layer in a five layer DBM:

![alt text](https://raw.githubusercontent.com/D-Niermann/Deep-Restricted-Boltzmann-Machine/master/Results/PCA%20Plots/PCA%20Methode/Layer_3/6.png "Generated images from a deep BM.")

It clusters very well as can be seen in the PCA. The histogram shows on what classes the neuron reacts the most.

In the first hidden layer, a typical neuron cannot cluster the classes so well, as seen here:
![alt text](https://raw.githubusercontent.com/D-Niermann/Deep-Restricted-Boltzmann-Machine/master/Results/PCA%20Plots/PCA%20Methode/Layer_1/7.png "Generated images from a deep BM.")
Most of the time, neurons in this layer can distinguish between certain classes alread as seen above, but they react to far to many classes to be usefull for classification. Thats why layter neuron sort out even more details and therefore distinguish only between one or two classes or, in extreme cases, filter out special cases within one class (e.g. for MNIST spezial methods of drawing a digit).

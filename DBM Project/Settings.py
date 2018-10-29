UserSettings = {
	# training process
	"N_BATCHES_PRETRAIN" : 20, 				# how many batches per epoch for pretraining
	"N_BATCHES_TRAIN"    : 20, 				# how many batches per epoch for complete DBM training
	"N_EPOCHS_PRETRAIN"  : [0,0,0,0,0,0], 	# pretrain epochs for each RBM
	"N_EPOCHS_TRAIN"     : 1, 				# how often to iter through the test images
	"TEST_EVERY_EPOCH"   : 1, 					# how many epochs to train before testing on the test data

	### learnrates
	"LEARNRATE_PRETRAIN" : 0.001 ,						# learnrate for pretraining
	"LEARNRATE_START"    : 0.01,						# starting learnrate
	"LEARNRATE_SLOPE"    : 1. ,							# bigger number -> smaller slope

	# freerunning steps (replaced in get_N() after first run)
	"N_FREERUN_START" : 2,						# how many freerunning steps to make (gets reset after 1 epoch in get_N() )


	### temperature
	"TEMP_START"    : 0.01	,			# starting temp
	"TEMP_SLOPE"    : 0 , #10e-7		# linear decrease slope higher number -> fast cooling
	"TEMP_MIN"      : 0.01 ,			# how low temp can fall at minimum

	# seed for random number generation, set None to have no seed 
	"SEED" : 2,							# random seed for tf and np

	### state vars
	"DO_PRETRAINING" : 0,		# if no pretrain then files are automatically loaded
	"DO_TRAINING"    : 0,		# if to train the whole DBM
	"DO_TESTING"     : 1,		# if testing the DBM with test data
	"DO_SHOW_PLOTS"  : 1,		# if plots will show on display - either way they get saved into saveto_path

	"DO_CONTEXT"    : 0,		# if to test the context
	"DO_GEN_IMAGES" : 0,		# if to generate images (mode can be choosen at function call)
	"DO_NOISE_STAB" : 0,		# if to make a noise stability test,

	"USE_DROPOUT"  : 1,		# if to use synnaptic failure while training
	"DROPOUT_RATE" : 2,		# multiplication of random uniform synaptic failure matrix (higher number -> less failure)

	"DO_NORM_W"    : 1,		# if to norm the weights and biases to 1 while training

	### saving and loading
	"DO_SAVE_TO_FILE"       : 0, 	# if to save plots and data to file
	"DO_SAVE_PRETRAINED"    : 0, 	# if to save the pretrained weights seperately (for later use)
	"DO_LOAD_FROM_FILE"     : 1, 	# if to load weights and biases from datadir + pathsuffix
	"PATHSUFFIX"            : "Mon_Jun__4_15-55-25_2018_[784, 225, 225, 225, 10] - ['original'] 15%", 
							#"Mon_Jun__4_15-55-25_2018_[784, 225, 225, 225, 10] - ['original'] 15%"
								#"Thu_Jun__7_16-21-28_2018_[784, 225, 225, 225, 10] - ['15cont4']"
	"PATHSUFFIX_PRETRAINED" : "Thu_Jun__7_13-49-25_2018",


	"DBM_SHAPE" : [	28*28,
					15*15,
					15*15,
					15*15,					
					10],

	## only used in DBM_context class
	"LAYERS_TO_CONNECT" : [],		# layer index of which layer to connect the context layer to (v2 label layer is always connected)
}

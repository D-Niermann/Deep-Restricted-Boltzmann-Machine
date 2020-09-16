
# -*- coding: utf-8 -*-
#### Imports
if True:
	print ("Starting")

	import matplotlib as mpl
	import os, time, sys


	workdir  = os.path.dirname(os.path.realpath(__file__))
	data_dir = os.path.join(workdir, "data")
	os.chdir(workdir)

	# should be removed
	OS = "Mac"

	import numpy as np
	import numpy.random as rnd
	import matplotlib.pyplot as plt
	# plt.style.use('classic')
	import tensorflow as tf
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
	from pandas import DataFrame, Series,read_csv
	from math import exp, sqrt, sin, pi, cos 
	
	## import own dependencies
	from Logger 		import *
	from RBM_Functions 	import *
	from DBM_Class 		import *
	from gen_gradients 	import * 
	from gen_driveData 	import *


	 # set the numpy print precision
	np.set_printoptions(precision=3)


	### import seaborn? ###
	if True:
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
		mpl.rcParams['xtick.minor.visible']  = True
		mpl.rcParams['ytick.minor.visible']  = True

		# plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		# plt.rcParams['image.cmap'] = 'coolwarm'
		# seaborn.set_palette(seaborn.color_palette("Set2", 10))

	log = Logger(True)


	time_now = time.asctime()
	time_now = time_now.replace(":", "-")
	time_now = time_now.replace(" ", "_")


#### Get the arguments list from terminal
additional_args = sys.argv[1:]


###########################################################################################################
#### Load User Settings ###

if len(additional_args)>0:
	log.out("Loading Settings from ", additional_args[0])
	Settings = __import__(additional_args[0])
else:
	import DefaultSettings as Settings

# rename 
UserSettings = Settings.UserSettings

# path were results are saved 
saveto_path = data_dir+"/"+time_now+"_"+str(UserSettings["DBM_SHAPE"])
if len(additional_args) > 0:
	saveto_path  += " - " + str(additional_args)

## open the logger-file
if UserSettings["DO_SAVE_TO_FILE"]:
	os.makedirs(saveto_path)
	log.open(saveto_path)


### modify the parameters with additional_args
if len(additional_args) > 0:
	first_param = int(additional_args[1]) #...
	# change settigns here
	log.out(UserSettings)
	


###########################################################################################################
#### Get test and train data  #####
n_visible = UserSettings["DBM_SHAPE"][0]
n_hidden  = UserSettings["DBM_SHAPE"][1]
# define a weight matrix
w = rnd.randn(n_visible, n_hidden)*0.5
# set small elemts to zero
w[np.abs(w)<0.45] = 0
# fig = plt.figure("Org Weights")
# plt.matshow(w,fig.number)
# plt.colorbar()

# calculate test and train data
train_data, train_label = generateDriveData(n_visible, n_hidden, 3000, w)
test_data, test_label   = generateDriveData(n_visible, n_hidden, 100, w)

###########################################################################################################
#### Create a DBM  #####

DBM = DBM_class(UserSettings = UserSettings,
				logger = log,
				workdir = workdir,
				saveto_path = saveto_path,
				liveplot = 0
)


# pretrain each RBM seperately 
DBM.pretrain(train_data)

# train the complete DBM with train data
DBM.train(train_data, test_data, train_label, test_label)

# test the DBM with test data
DBM.test(test_data, test_label, N = 50)

# Plot the results or save them to file (see Settings.py)
DBM.plot_layer_div()
DBM.plot_train_errors()

DBM.show()


"""
- richtige daten versuchen zu lernen, zB die aus dem paper von alex

- gibbs sampling? generate images from freerunning ... grade sowas wie 
		"ich habe nur zwei von 5 infos, wie unsicher ist er und was könnten die anderen 3 infos sein"
		-> oder recommender: gebe niedrige unsicherheit vor, halte features fest die grade nicht geändert 
		werden können (wegen äußeren umständen) und dann sample die restlichen nodes

- split show results into multiple functions ? (show weights, show train log, show test restuls ....) 
		also check if testing again and then plotting again really plots the new tested daata and results


- change the zip(range, range) lines so that last part is not missing all the time 
"""

print("Finished")
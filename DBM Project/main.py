"""

import DBM_class

load data 

DBM = DBM_class(data, ...)


DBM.pretrain(...)

DBM.train(...)

DBM.test(...)

DBM.plot_results(what to plot)

"""


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
	try:
		first_param = int(additional_args[1])
		if len(additional_args) == 3:
			second_param = int(additional_args[2])
		log.out("Additional Arguments:", first_param, second_param)
		UserSettings["TEMP_START"] = second_param
		UserSettings["TEMP_MIN"] = second_param
		log.out(UserSettings)
	except:
		log.out("ERROR: Not using additional args!")



# load UserSettings into globals
log.out("For now, copying UserSettings into globals()")
for key in UserSettings:
	globals()[key] = UserSettings[key]

# better name for subset:
subset = SUBSPACE



###########################################################################################################
#### Get test and train data  #####
n_visible = DBM_SHAPE[0]
n_hidden = DBM_SHAPE[1]
# define a weight matrix
w = rnd.randn(n_visible, n_hidden)*0.5
# set small elemts to zero
w[np.abs(w)<0.45] = 0

# calculate test and train data
train_data, train_label = generateDriveData(n_visible, n_hidden, 2000, w)
test_data, test_label  = generateDriveData(n_visible, n_hidden, 200, w)

###########################################################################################################
#### Create a DBM  #####

DBM = DBM_class(shape = DBM_SHAPE,
				liveplot = 0,
				classification = DO_CLASSIFICATION,
				UserSettings = UserSettings,
				logger = log,
				workdir = workdir,
				saveto_path = saveto_path
)


# pretrain each RBM seperately 
DBM.pretrain(train_data)

# train the complete DBM with train data
DBM.train(train_data, test_data, train_label, test_label)

# test the DBM with test data
DBM.test(test_data, test_label, N = 50)

# Plot the results or save them to file (see Settings.py)
DBM.show_results()

# gibbs sampling? generate images from freerunning ... grade sowas wie "ich habe nur zwei von 5 infos, wie unsicher ist er und was könnten die anderen 3 infos sein"
# split show results into multiple functions ? (show weights, show train log, show test restuls ....) also check if testing again and then plotting again really plots the new tested daata and results
# include "is image data" variable in init to create better plots and analytics -> show matri tiled or not, show v2 layer desired vs result, better remap of data to images fro plot

print("Finished")
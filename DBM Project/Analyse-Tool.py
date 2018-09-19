if True:
	print ("Starting")

	import matplotlib as mpl
	import os,time,sys

	try: # if on macbook
		workdir="/Users/Niermann/Google Drive/Masterarbeit/Python/DBM Project"
		os.chdir(workdir)
		import_seaborn = True
		OS = "Mac"
	except: #else on university machine
		workdir="/home/dario/Dokumente/DBM Project"
		os.chdir(workdir)
		mpl.use("Agg") #use this to not display plots but save them
		import_seaborn = True
		OS = "Linux"
		import importlib

	data_dir=workdir+"/data/"

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

	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
		# plt.rcParams['image.cmap'] = 'coolwarm'
		# seaborn.set_palette(seaborn.color_palette("Set2", 10))

	log=Logger(True)

	from tensorflow.examples.tutorials.mnist import input_data
	time_now = time.asctime()
	time_now = time_now.replace(":", "-")
	time_now = time_now.replace(" ", "_")
####################################################################################

def load_firerates(folder_name, load_context):

	test_data_dir = data_dir+folder_name+"/FireratesTest/"
	c_data_dir    = data_dir+folder_name+"/FireratesContext/"
	nc_data_dir   = data_dir+folder_name+"/FireratesNoContext/"

	f_test = []
	f_c    = []
	f_nc   = []


	log.start("Loading Test Firerates")
	for f in os.listdir(test_data_dir):
		f_test.append(np.loadtxt(test_data_dir+f))
	log.end()

	if load_context:
		log.start("Loading Context Firerates")
		for f in os.listdir(c_data_dir):
			f_c.append(np.loadtxt(c_data_dir+f))
		for f in os.listdir(nc_data_dir):
			f_nc.append(np.loadtxt(nc_data_dir+f))
		log.end()
		
		
	return f_test,f_c,f_nc


####################################################################################
# User Settings
####################################################################################

folder_name = "Tue_Sep_18_11-28-06_2018_[784, 225, 225, 225, 10]/"
load_data    = 0
load_context = 1


####################################################################################
# LOADING
####################################################################################
#load the logfile
logfile = load_logfile(data_dir+folder_name)

# load the firerates
if load_data:
	f_test, f_c, f_nc = load_firerates(folder_name, load_context)

# load the weights 
w = []
for i in range(len(logfile["DBM_SHAPE"])-1):
	w_str = data_dir+logfile["PATHSUFFIX"]+"/w%i.txt"%i
	w.append(np.loadtxt(w_str))


####################################################################################
# Analyzing
####################################################################################


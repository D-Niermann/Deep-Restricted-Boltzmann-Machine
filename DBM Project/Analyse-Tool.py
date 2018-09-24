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

if "train_data" not in globals():	
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	train_data, train_label, test_data, test_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

####################################################################################
# Functions
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

def get_neuron_subdata(neuron,firerates,fire_thresh,context):
	""" 
	gets the images with corresponding labels on which the given neuron fired 
	more often than fire_thresh.
	subdata contains the full image vectos.
	label contains the true label value (from 0-9) and not the vectors.
	context [bool] :: weather context data was used or the full test data
	"""

	neurons = np.where(firerates[:,neuron]>fire_thresh)[0]

	if len(neurons)>1:
		if context==False:
			subdata     = test_data[neurons]
			sublabel    = test_label[neurons]
		else:
			subdata     = test_data[index_for_number_gibbs[:]][neurons]
			sublabel    = test_label[index_for_number_gibbs[:]][neurons]


		# label = []
		# for i in range(len(sublabel)):
		label = np.where(sublabel==1)[1]
			# label.append(where)
		# label = np.array(label).astype(np.float)
		
		return subdata,label
	else:
		return None,None
####################################################################################
# User Settings
####################################################################################

folder_name = "Tue_Sep_18_11-28-06_2018_[784, 225, 225, 225, 10]/"
load_test_data    = 0
load_context_data = 0


####################################################################################
# LOADING
####################################################################################
#load the logfile
logfile = load_logfile(data_dir+folder_name)

# load the firerates
if load_test_data:
	f_test, f_c, f_nc = load_firerates(folder_name, load_context_data)

# load the weights 
w = []
for i in range(len(logfile["DBM_SHAPE"])-1):
	w_str = data_dir+logfile["PATHSUFFIX"]+"/w%i.txt"%i
	w.append(np.loadtxt(w_str))


####################################################################################
# Calculations
####################################################################################
n_layers = len(logfile["DBM_SHAPE"])
unit_diversity_test   = [None]*n_layers
unit_information_test = [None]*n_layers
unit_entropy_test     = [None]*n_layers
f_test_mean           = [None]*n_layers

for l in range(len(f_test_mean)):
	# average
	f_test_mean[l]           = np.mean(f_test[l], axis = 0)
	# standard deviation
	unit_diversity_test[l]   = np.sqrt(np.mean(np.square(f_test[l] - f_test_mean[l]),axis=0))
	# information
	unit_information_test[l] = -np.log(f_test_mean[l])
	### entropy                = > expected information: prob * -log(prob) + anti_prob * -log(anit_prob)
	# calc summands
	a = f_test_mean[l]*-np.log(f_test_mean[l])
	b = (1-f_test_mean[l])*np.log(1-f_test_mean[l])
	# clear nans 
	a[np.isnan(a)] = 0
	b[np.isnan(b)] = 0
	# add up 
	unit_entropy_test[l] = a - b

### if found neuron, check how it fired over all images
neuron_ind  = 140
layer       = 3
fire_thresh = 0.8
plt.figure()
plt.hist(f_test[layer][:,neuron_ind])

## PCA
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
pca            = PCA(2)
subdata, label = get_neuron_subdata(neuron_ind,f_test[layer],fire_thresh,False)
pca.fit(subdata)
trans          = pca.transform(subdata)

"""
## nur noch neurone finden die sich krass ändern bei c/nc und dann die 
## PCA analyse da drauf kloppen und ein paar schöne bilder machen
"""
####################################################################################
# Plotting
####################################################################################

# plot means over layer
fig,ax = plt.subplots(1,n_layers-2, figsize=(8,3),sharey="row")
for l in range(1,n_layers-1):
	ax[l-1].hist(unit_entropy_test[l], edgecolor="k", lw = 0.5)
	ax[l-1].set_xlabel("Entropy of %s"%get_layer_label("DBM", n_layers, l))
ax[0].set_ylabel(r"$N$")
plt.tight_layout()

#

ent_means = []
label = []
for l in range(1,n_layers-1):
	ent_means.append(np.mean(unit_entropy_test[l]))
	label.append(get_layer_label("DBM", n_layers, l))
plt.figure("Entropy means over layer")
plt.bar(range(1,n_layers-1),ent_means)
plt.ylabel("Average Entropy")
plt.xticks(range(1,n_layers-1),label)
	


## plot the pca things
fig,ax = plt.subplots(1,4, figsize=(10,2.5))
for k in range(2):
	ax[k].matshow(pca.components_[k].reshape(28,28))
	ax[k].set_yticks([])
	ax[k].set_xticks([])
	ax[k].set_title("Expl. Variance: "+str(round(pca.explained_variance_ratio_[k],3)))
mapp3 = ax[2].scatter(trans[::2,0], trans[::2,1], c = label[::2],cmap=plt.cm.get_cmap('gist_rainbow', 10),alpha=0.5,vmin=0,vmax=9)
ax[2].set_title("PCA of "+r"$V^{(s)}$")
ax[2].set_xlabel("1. PC")
ax[2].set_ylabel("2. PC")
plt.colorbar(ax = ax[2], mappable = mapp3)
ax[3].hist(label,edgecolor = "k", lw = 0.3)
ax[3].set_xticks(range(10))
ax[3].set_xlabel("Class")
ax[3].set_title("Histogram of "+r"$V^{(s)}$")
plt.tight_layout()

plt.show()
print ("Starting...")
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
# import scipy.ndimage.filters as filters
# import pandas as pd
import os,time,sys
from math import exp,sqrt,sin,pi,cos,log
np.set_printoptions(precision=3)

workdir="/Users/Niermann/Google Drive/Masterarbeit/Python/DBM Project"
# workdir="/home/dario/Dokumente/DBM Project"

data_dir=workdir+"/Plot_data"	
os.chdir(workdir)
from Logger import *
from RBM_Functions import *
os.chdir(data_dir)
### import seaborn? ###
if 1:
	import seaborn

	seaborn.set(font_scale=1.3)
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

########################################################################################

####################################################
### plot the class errors for 60 epochs 
### no pretraining, 2 steps gibbs sampling, shape = [784,400,10]
index = [0.001,0.005,0.01,0.05,0.1,0.5,1]
class_errors_l_t = np.loadtxt("class errors learnrate over temp.txt")

seaborn.heatmap((class_errors_l_t),annot=True,cmap="RdYlBu_r")
plt.xlabel("Temperature")
plt.ylabel("Learnrate")
plt.title("Classification Error")
plt.yticks(np.add(range(7),0.5),index)
plt.xticks(np.add(range(7),0.5),index)
####################################################



plt.show()
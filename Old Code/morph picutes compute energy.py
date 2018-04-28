#### Imports
# print "Starting"
if 0:
	# -*- coding: utf-8 -*-
	print "Starting..."
	import numpy as np
	import numpy.random as rnd
	import matplotlib.pyplot as plt
	import matplotlib.image as img
	import tensorflow as tf
	# import scipy.ndimage.filters as filters
	# import pandas as pd
	import os,time,sys
	from math import exp,sqrt,sin,pi,cos,log
	np.set_printoptions(precision=3)
	# plt.style.use('ggplot')

	### import seaborn? ###
	if 0:
		import seaborn

		seaborn.set(font_scale=1.4)
		seaborn.set_style("ticks",
			{
			'axes.grid':            1,
			'grid.linestyle':       u':',
			'legend.numpoints':     1,
			'legend.scatterpoints': 1,
			'axes.linewidth':       1,
			'xtick.direction':      'in',
			'ytick.direction':      'in',
			'xtick.major.size': 	5,
			'xtick.minor.size': 	1.0,
			'legend.frameon':       True,
			'ytick.major.size': 	5,
			'ytick.minor.size': 	1.0
			})

	mpl.rcParams["image.cmap"] = "jet"
	mpl.rcParams["grid.linewidth"] = 0.5
	mpl.rcParams["lines.linewidth"] = 1.
	mpl.rcParams["font.family"]= "serif"
	# plt.rcParams['image.cmap'] = 'coolwarm'


# visuni=10
# class Player(object):
# 	# global visuni
# 	def __init__(self,x,y):
# 		self.x = x
# 		self.y = y
# 		x      = 1

# 	def sum(self):
# 		print self.x
# 		return self.x+self.y+visuni

# p1=Player(1, 2)
# a=tf.Variable([1,2,3],name="asdasd")
# with tf.Session() as sess_test:
# 	sess_test.run(tf.global_variables_initializer())
# 	print a.eval(session=sess_test)

# Player.visuni=0
def sample(x):
	s=np.shape(x)
	rng=rnd.random(s)
	x=x>=rng
	return np.array(x)


fig,ax=plt.subplots(2,1)
data0=ax[0].matshow(a.reshape(28,28))
data1,=ax[1].plot([], [], linewidth=1)
energy1=[]
ax[1].set_ylabel("Energy")
c_save=[]

switch=0

n=10 #number of pictures
m=range(n*100)

for j in range(50,50+n):
	a=np.copy(test_data[j])
	b=np.copy(test_data[j+1])
	c=np.copy(a)
	
	for i in range(100):
		ax[1].cla()
		c-=a*0.01
		c+=(b*0.01)
		h1=sigmoid_np(np.dot(c,DBM.w1_np)+DBM.bias2_np,1.)
		energy1.append(-np.dot(c, np.dot(DBM.w1_np,h1.T)))
		if i>1:
			if energy1[-1]>energy1[-2] and switch==0:
				c_save.append(c)
				switch=1
			if energy1[-1]<energy1[-2]:
				switch=0
		# c=sample(c)
		# c*=1./c.max()
		# c=np.round(c)
		if plt.fignum_exists(fig.number):
			data0.set_data(sample(c).reshape(28,28))
			ax[1].plot(m[0:len(energy)],energy)
			ax[1].set_xlim(0,len(m))
			plt.pause(0.02)


# plt.close()


plt.plot(np.linspace(0,10,len(energy)),energy1,label="500 Epochs")
plt.plot(np.linspace(0,10,len(energy)),energy,label="20 Epochs")
plt.xlabel("Picture Index")
plt.ylabel("Energy "+r"$E$")
plt.legend(loc="best")
fig,ax=plt.subplots(1,10)
ax[0].matshow(test_data[50].reshape(28,28))
ax[0].grid(False)
ax[i].set_xticks([])
ax[i].set_yticks([])

for i in range(len(c_save)):
	ax[i].matshow(np.reshape(c_save[i],[28,28]))
	ax[i].set_xticks([])
	ax[i].set_yticks([])
	ax[i].grid(False)


plt.show()



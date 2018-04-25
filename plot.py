#### Plot
# Plot the Weights, Errors and other informations
print("Starting")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import seaborn
import pandas as pd

#workdir   = "/Users/Niermann/Google Drive/Masterarbeit/Python/DBM Project"
workdir = "/home/dario/Dokumente/DBM Project"
if 1:
	import seaborn

	seaborn.set(font_scale=1.0)
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
	# seaborn.set_palette("Set1", 8, .75)
	mpl.rcParams["image.cmap"] = "jet"
	mpl.rcParams["grid.linewidth"] = 1
	mpl.rcParams["lines.linewidth"] = 1.25
	mpl.rcParams["font.family"]= "serif"
os.chdir(workdir)
from Logger import *
from RBM_Functions import *
log=Logger(True)


##########################################################################################
class container(object):
	def load_files(self,directory,files_to_load):
		os.chdir(data_dir+directory)
	
		#log.info("Changed dir to: ",directory)
		for files in os.listdir(os.getcwd()):
			if files!="logfile.txt" and files in files_to_load:
				#log.out("Loading ",files)
				self.__dict__[(files[:-4])]=np.loadtxt(files)
			
			if files=="logfile.txt":
				logfile_=[]
				with open("logfile.txt","r") as logfile:
					for line in logfile:
						for i in range(len(line)):
							if line[i]==",":
								save=i
						value=line[save+1:-1]
						try:
							value=float(value)
						except:
							pass
						logfile_.append([(line[0:save]),value])
				for i in range(len(logfile_)):
					key=logfile_[i][0]
					value=logfile_[i][1]
					self.__dict__[key]=value

			if files == "save_dict.csv":
				self.__dict__[(files[:-4])]= pd.read_csv(files)
			
			
def load_data(data_dir):
	conts=[]
	os.chdir(data_dir)
	folders = os.listdir(os.getcwd())
	
	for folder in folders:

		if " - " in folder:
			conts.append(container())
			conts[-1].load_files(folder,files_to_load)
			conts[-1].name=folder
			#if conts[-1].n_units_second_layer==1369:
		#	conts[-1].load_files(folder,["w1.txt"])
	
	return conts


##########################################################################################

files_to_load = ["Classification Error on test images.txt",
				"Classification_Error_on_test_images.txt",
				"Recon_Error_on_test_images.txt"]

#### which ones to plot
plot_errors_temp_learnrate = 1
plot_inits_same            = 0
plot_freesteps             = 0
plot_temp_learn_slopes     = 0
plot_increasing_temp       = 0
plot_tslope_lslope         = 0

##########################################################################################


#### Plot 
log.info("Plotting")

if plot_errors_temp_learnrate:

	
	index = np.array([0.001,0.005,0.01,0.05,0.1,0.5,1,1.5,2,2.5,3,3.5,4,4.5])
	errors = np.zeros([len(index),len(index)])
	errors_full = np.zeros([len(index),len(index),30])
	mins = np.zeros([len(index),len(index)])

	data_dir  = workdir+"/data/learnrate-temp/"
	conts = load_data(data_dir)

	data_dir  = workdir+"/data/learnrate-temp/new method/"
	conts_n = load_data(data_dir)
	
	fig,ax=plt.subplots(1,len(index),sharey="row")

	## iter through old method conts
	for i in range(len(conts)):
		error_i = conts[i].__dict__["Classification Error on test images"]
		temp_i = conts[i].Temperature
		l_i = conts[i].learnrate_dbm_train
		x = np.where(index == l_i)[0]
		y = np.where(index == temp_i)[0]

		errors[x,y] = error_i[-1]
		errors_full[x,y] = error_i
		mins[x,y] = conts[i].__dict__["Classification Error on test images"].min()
		x = x[0]

	## iter though new mthod conts
	for i in range(len(conts_n)):
		sd = conts_n[i].save_dict
		epochs = sd["Test_Epoch"].values[sd["Test_Epoch"].notna()]
		error_i = sd["Class_Error"].values[sd["Class_Error"].notna()]
		temp_i = conts_n[i].starting_temp
		l_i = conts_n[i].learnrate_dbm_train
		x = np.where(index == l_i)[0]
		y = np.where(index == temp_i)[0]
		
		errors[x,y] = error_i[-1]
		error_i=np.concatenate([error_i,np.zeros(19)])
		errors_full[x,y] = error_i
		mins[x,y] = error_i.min()
		x = x[0]
		
	
	for i in range(len(index)):	
		for j in range(len(index)):
			color = next(ax[i]._get_lines.prop_cycler)['color']
			ax[i].plot(range(0,60,2),errors_full[i][j],"-",lw=1,alpha = 1, color=color,label=r"$T$ ="+" "+str(index[j]))
		#ax.plot(range(0,60,2),conts[i].__dict__["Recon Error on test images"],"-",color=color,label=conts[i].name[-7:-1])

		ax[i].set_xlabel("Epoch")
		if i==0:
			ax[i].set_ylabel("Error")
		ax[i].set_title("$\eta$ ="+str(index[i]))
		ax[i].grid(True)

	ax[-1].legend(loc='center left',bbox_to_anchor=(1.1,0.5),ncol=1)
		

	plt.figure()
	seaborn.heatmap(errors,annot=True,cmap="RdYlBu_r")
	# plt.colorbar()
	plt.ylabel("Learnrate")
	plt.xlabel("Temperature")
	plt.title("Classification Error after 60 epochs")
	plt.xticks(np.add(range(len(index)),0.5),index)
	plt.yticks(np.add(range(len(index)),0.5),index)
	min_pos = np.where(errors==errors.min())
	
	# plt.matshow(np.abs(errors-mins),cmap="RdYlBu_r")


if plot_inits_same:
	""" 
	plot bei dem alle parameter die selben sind, nur die initial values von den gewichten wurden
 	verandert """
	log.out("plot inits same")
	
	data_dir  = workdir+"/data/Init_with_all_same/"
	conts = load_data(data_dir)
	data_dir  = workdir+"/data/Init_with_all_same/new_method/"
	conts_new_m = load_data(data_dir)
	inits_errors = []
	inits_errors_n = {"1e-6":[], "1e-2" : [], "0":[]}
	for i,c in enumerate(conts):
		
		inits_errors.append(c.__dict__["Classification_Error_on_test_images"])
		recon_error_i = c.__dict__["Recon_Error_on_test_images"]

	for i,c in enumerate(conts_new_m):
		suffix = c.name[-6:-2]
		try:
			inits_errors_n[suffix].append(c.save_dict["Class_Error"][:11].values)
		except:
			suffix = "0"
			inits_errors_n[suffix].append(c.save_dict["Class_Error"][:11].values)

		#plt.plot(inits_errors[:,0],inits_errors[:,1])

	inits_errors = np.array(inits_errors)

	fig,ax = plt.subplots(2,1)
	seaborn.tsplot(data=inits_errors[:,:,1],ax=ax[1],time=range(0,22,2),err_style = "unit_traces",interpolate=False)
	seaborn.tsplot(data=inits_errors[:,:,1],ax=ax[1],time=range(0,22,2),err_style = "ci_bars",ci=[100],interpolate=False)
	seaborn.tsplot(data=inits_errors[:,:,1],ax=ax[1],time=range(0,22,2),ci=[100],interpolate=False,condition="1e-4")
	color = next(ax[1]._get_lines.prop_cycler)['color']
	

	for suffix in inits_errors_n:
		log.out(suffix)
		color = next(ax[1]._get_lines.prop_cycler)['color']
		ax_i = ax[1]
		
		if suffix == "0":
			ax_i = ax[0]
		
		seaborn.tsplot(data=inits_errors_n[suffix][:],ax=ax_i,color=color,time=range(0,22,2),err_style = "unit_traces",interpolate=False)
		seaborn.tsplot(data=inits_errors_n[suffix][:],ax=ax_i,color=color,time=range(0,22,2),err_style = "ci_bars",ci=[100],interpolate=False)
		seaborn.tsplot(data=inits_errors_n[suffix][:],ax=ax_i,color=color,time=range(0,22,2),ci=[100],interpolate=False,condition=suffix)
		

	plt.legend()
	ax[0].set_ylabel("Error")
	plt.xlabel("Epoch")
	plt.ylabel("Error")


if plot_freesteps:
	log.out("plot freerunning")
	plt.figure("Freerunning_Steps")
	data_dir  = workdir+"/data/Freerunning_Steps/"
	conts = load_data(data_dir)
	os.chdir(data_dir)
	folders = os.listdir(os.getcwd())
	n_steps=[]
	for f in folders:
		print(f)
		file = open(f+"/"+"Logger-File.txt","r")
		for line in file:
			if "Freerunning for" in line:
				save_line = line
				break
		n_steps.append(save_line[-10:-8])

	os.chdir(workdir)
	for i,c in enumerate(conts):
	
		error_i = c.__dict__["Classification Error on test images"]
		
		plt.plot(error_i,label=n_steps[i])
	
	plt.xlabel("Epoch")
	plt.ylabel("Error")
	plt.legend(loc="best")


if plot_temp_learn_slopes:
	fig,ax=plt.subplots(2,1)
	fig_2,ax_2=plt.subplots(2,1)
	data_dir = workdir+"/data/Temp_Learn_Slopes/"
	conts = load_data(data_dir)
	conts = sorted(conts,key=lambda x: x.learnrate_dbm_slope)
	for c in conts:
		if c.name[-15:]!="high_temp_slope":
			slope = c.learnrate_dbm_slope
			epochs = c.save_dict["Test_Epoch"]
			errors = c.save_dict["Class_Error"]
			learn = c.save_dict["Learnrate"].values
			learn = learn.tolist()
			learn.insert(0, 0.01)
			color = next(ax[0]._get_lines.prop_cycler)['color']
			ax[0].plot(learn,"o-",color=color,label=slope,markersize=5 )
			ax[1].plot(epochs,errors,"o-",color=color,label=slope ,markersize=5 )
			ax[0].set_title("Constant Temperature")
		else:
			slope = c.learnrate_dbm_slope
			epochs = c.save_dict["Test_Epoch"]
			errors = c.save_dict["Class_Error"]
			learn = c.save_dict["Learnrate"].values
			learn = learn.tolist()
			learn.insert(0, 0.01)
			color = next(ax_2[0]._get_lines.prop_cycler)['color']
			ax_2[0].plot(learn,"o-",color=color,label=slope,markersize=5)
			ax_2[1].plot(epochs,errors,"o-",color=color,label=slope ,markersize=5)
			ax_2[0].set_title("Increasing Temperature")

	ax[0].legend(loc="center left",bbox_to_anchor = (1.,0.5))
	ax[1].legend(loc="center left",bbox_to_anchor = (1.,0.5))
	ax[0].set_xticks(range(0,21,2))
	ax[1].set_xticks(range(1,22,2))


if plot_increasing_temp:
	plt.figure()
	data_dir = workdir+"/data/increasing_temp/"
	conts = load_data(data_dir)
	conts = sorted(conts,key=lambda x: x.temp_slope)

	for c in conts:
		ts = c.temp_slope
		epochs = c.save_dict["Test_Epoch"]
		errors = c.save_dict["Class_Error"]

		plt.plot(epochs,errors,label=ts)
	plt.legend()


if plot_tslope_lslope:
	fig,ax      = plt.subplots(1, 6, sharey ="row")
	data_dir    = workdir + "/data/Tslope_Lslope/"
	conts       = load_data(data_dir)
	conts       = sorted(conts,key=lambda x: x.learnrate_dbm_slope)
	errors_grid = np.zeros([6,6])
	slopes      = np.array([0.005, 0.01, 0.05, 0.1, 0.5, 1.0])

	for c in conts:

		# get vars from container
		learnrate_slope = c.learnrate_dbm_slope
		temp_slope      = c.temp_slope
		epochs          = c.save_dict["Test_Epoch"].values[c.save_dict["Test_Epoch"].notna()]
		class_error     = c.save_dict["Class_Error"].values[c.save_dict["Class_Error"].notna()]
		recon_error     = c.save_dict["Recon_Error"].values[c.save_dict["Recon_Error"].notna()]
		temp = c.save_dict["Temperature"].values
		learn = c.save_dict["Learnrate"].values

		# look where in the grid to place the error
		x = np.where(slopes == learnrate_slope)[0][0]
		y = np.where(slopes == temp_slope)[0][0]
		
		# fill last error value in grid
		errors_grid[x,y]=class_error.min()

		ax[y].plot(epochs,class_error,lw=1,label = r"$\eta_s$ = " +str(c.learnrate_dbm_slope))
	
		if x==1.0:
			ax[y].set_title("T_slope = " + str(temp_slope))
			ax[y].grid(True)
			ax[y].set_xticks(range(0,70,25))
			if y==0:
				ax[y].set_ylabel("Classification Error")
			ax[y].set_xlabel("Epoch")


	ax[-1].legend(loc="center left",bbox_to_anchor=(1,0.5))

	plt.figure()
	seaborn.heatmap(errors_grid,annot=True,cmap="RdYlBu_r")
	plt.ylabel("Learnrate Slope")
	plt.xlabel("Temperature Slope")
	plt.xticks(np.add(range(6),0.5),slopes)
	plt.yticks(np.add(range(6),0.5),slopes)


plt.show()
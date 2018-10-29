# from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

kmeans = KMeans(n_clusters=1)
pca = PCA(2)
mean_img = np.mean(train_data[:10000],0)

## user variables
layer       = 3
fire_thresh = 0.8
w_thresh    = 0.04
min_mean    = 0.05
max_mean    = 0.8


# os.chdir("/Users/Niermann/Desktop/Plots/Layer_%i"%layer)
means_c = [None]*DBM.n_layers
means_nc = [None]*DBM.n_layers
# for l in range(1,DBM.n_layers-1):
# 	means_c[l] = np.mean(DBM.firerate_c[l],0)
# 	means_nc[l] = np.mean(DBM.firerate_nc[l],0)

# fig,ax=plt.subplots(3,1,sharex="col")
# for l in range(1,4):
# 	ax[l-1].hist(means[l],bins=30,alpha=0.3)
# 	ax[l-1].set_title("Average Fire Rate Layer %i"%l)
"""
neuron_number_ = []
neuron_means_ =[]
for neuron in range(DBM.SHAPE[layer]):
	# m_c = means_c[layer][neuron]
	# m_nc = means_nc[layer][neuron]

	if m>min_mean and m<max_mean:
		neuron_number_.append(np.where(means_c[layer]==m_c)[0][0])
		neuron_means_.append(m)
	if len(neuron_number_)>20:
		break
"""
neuron_number_ = list(set(neuron_number_))
neuron_number_ = [38,81,99,112,114]

## get neurons based on their weight strength
# neuron_number_ = np.where(np.abs(DBM.w_np[-1][:,5:])>w_thresh)[0]
# target_class = np.where(np.abs(DBM.w_np[-1][:,5:])>w_thresh)[1]

# # get neurons by max weights from weights to context layer
# neuron_number_ = ind[0]



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
hist_diff  = np.zeros(255)
for neuron in range(len(neuron_number_)):
	
	neuron_number  = neuron_number_[neuron]
	log.out("Neuron:" , neuron_number)


	subdata,label = get_neuron_subdata(neuron_number,DBM.firerate_test[layer][:,:],fire_thresh,False)
	# subdata_c,label_c = get_neuron_subdata(neuron_number,DBM.firerate_c[layer],fire_thresh,True)
	# subdata_nc,label_nc = get_neuron_subdata(neuron_number,DBM.firerate_nc[layer],fire_thresh,True)
	try:
		if label_c==None:
			log.info("Skipped")
			continue
	except:
		pass
	# hist_label_c    = np.histogram(label_c,bins=len(subspace))[0]
	# hist_label_nc   = np.histogram(label_nc,bins=len(subspace))[0]

	"""
	hist_diff[neuron]=np.mean(hist_label_nc - hist_label_c)
	if np.abs(hist_diff[neuron])>20:
		plt.figure()
		plt.bar(subspace,hist_label_c,alpha=0.5,label="context")
		plt.bar(subspace,hist_label_nc,alpha=0.5,label="no context")
		plt.legend(loc="best")
	"""

	mean_subdata = np.mean(subdata,0)
	pca.fit(subdata)
	trans = pca.transform(subdata)
	# kmeans.fit(subdata)
	


	fig,ax = plt.subplots(1,5,figsize=(13,2.9))
	fig.suptitle("Layer %i | Neuron %i"%(layer,neuron_number))#+ r" $<f>$ = %f" %(neuron_means_[neuron]))

					## old code for plotting many PCA compnents or kmeans 
					# k = 0
					# for i in range(2):
					# 	for j in range(2):
							
					# 		# mapp = ax[i,j].matshow(pca.components_[k].reshape(28,28))
					# 		# mapp = ax[i,j].matshow(kmeans.cluster_centers_[k].reshape(28,28))
					# 		plt.colorbar(ax=ax[i,j], mappable=mapp)
					# 		# ax[i,j].set_title("Expl. variance = "+str(round(pca.explained_variance_ratio_[k],3)))
					# 		ax[i,j].set_xticks([])
					# 		ax[i,j].set_yticks([])
					# 		k+=1

	ax[0].set_title(r"$<V_i^{(s)}>-<V_i>$")
	mapp = ax[0].matshow((mean_subdata-mean_img).reshape(28,28))
	plt.colorbar(ax=ax[0], mappable=mapp)
	ax[0].set_xticks([])
	ax[0].set_yticks([])
	
	if layer==1:
		ax[1].set_title("Corresponding Filter")
		mapp2 = ax[1].matshow(DBM.w_np[0][:,neuron_number].reshape(28,28))
		plt.colorbar(ax=ax[1], mappable=mapp2)
		ax[1].set_xticks([])
		ax[1].set_yticks([])
	else:
		ax[1].set_title("Firerates")
		ax[1].bar(1,np.mean(DBM.firerate_test[layer][:,neuron_number]))
		# ax[1].bar(2,np.mean(DBM.firerate_c[layer][:,neuron_number]))
		# ax[1].bar(3,np.mean(DBM.firerate_nc[layer][:,neuron_number]))
		ax[1].set_xticks([1,2,3])
		ax[1].set_xticklabels(["Test","Context","No Context"],rotation=20)
					### old k means plot 
					# mapp2 = ax[1].matshow(kmeans.cluster_centers_[0].reshape(28,28))
					# plt.colorbar(ax=ax[1], mappable=mapp2)
					# ax[1].set_xticks([])
					# ax[1].set_yticks([])



	mapp3 = ax[2].scatter(trans[::2,0], trans[::2,1], c = label[::2],cmap=plt.cm.get_cmap('gist_rainbow', 10),alpha=0.5,vmin=0,vmax=9)
	ax[2].set_title("PCA of "+r"$V^{(s)}$")
	plt.colorbar(ax = ax[2], mappable = mapp3)
	ax[2].set_ylabel("")
	ax[2].set_xlabel("")


	
	ax[3].set_title("Histogram of "+r"$V^{(s)}$")
	
	hist_label_test = np.histogram(label,bins=10)[0]


	ax[3].bar(range(10),  hist_label_test, lw=0.2, edgecolor="k", alpha=0.7, label="Test")
	# ax[3].bar(subspace,  hist_label_c,    lw=0.2, edgecolor="k", alpha=0.5, width=0.6, label="Context")
	# ax[3].bar(subspace,  hist_label_nc,   lw=0.2, edgecolor="k", alpha=0.5, width=0.2, label="No Context")
	ax[3].set_xticks(range(10))
	ax[3].set_ylabel("N")
	ax[3].set_xlabel("Class")
	# ax[3].legend(loc="best")

	

	ax[4].set_title("Weight strength")
	ax[4].bar(range(10),DBM.w_np[-1][neuron_number,:])
	ax[4].set_xticks(range(10))
	# plt.savefig(str(neuron),dpi=250)
	# mapp = ax[1,1].matshow(pca.components_[2].reshape(28,28))
	# plt.colorbar(ax=ax[1,1], mappable=mapp)

	plt.subplots_adjust(top=0.65, bottom = 0.19, wspace=0.4, left=0.04, right=0.97)
	            # wspace=None, hspace=None)


log.reset()
plt.show()
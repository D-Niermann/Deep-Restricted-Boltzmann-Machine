# from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

kmeans = KMeans(n_clusters=4)
pca = PCA(4)

## user variables
layer       = 2
fire_thresh = 0.9
min_mean    = 0.8
max_mean    = 0.96


# os.chdir("/Users/Niermann/Desktop/Plots/Layer_%i"%layer)
means = [None]*DBM.n_layers
for l in range(DBM.n_layers):
	means[l] = np.mean(DBM.layer_save_test[l][-1,:,:],0)

fig,ax=plt.subplots(3,1,sharex="col")
for l in range(1,4):
	ax[l-1].hist(means[l],bins=30,alpha=0.3)
	ax[l-1].set_title("Average Fire Rate Layer %i"%l)

neuron_number_ = []
neuron_means_ =[]
for i in range(DBM.SHAPE[layer]):
	m = means[layer][i]
	if m>min_mean and m<max_mean:
		neuron_number_.append(np.where(means[layer]==m)[0][0])
		neuron_means_.append(m)
	if len(neuron_number_)>10:
		break

neuron_number_ = list(set(neuron_number_))

# neuron_number_ = [3]

def calc_neuron_hist(layer, neuron_index, neuron_data, neuron_label, fire_thresh):
	""" calcs the ...
	layer 			:: in which layer the neurons are
	neuron_index 	:: array of neurons to compute the hist for
	neuron_data 	:: where the activities (averages) for each image are stored,
						e.g. DBM.layer_save_test, shape [batchsize,neurons]
	neuron_label 	:: corresponding labels to neuron_data, e.g. test_label or 
						subsets of labels for context
	"""
	hist = [None]*len(neuron_index)
	for n in range(len(neuron_index)):

		# find images that have high fire rates for that neuron 
		w = np.where(neuron_data[:,n]>fire_thresh)[0]

		# get the corresponding label to the found images
		sublabel = neuron_label[w]
		
		# create a real label vector where not the [[0,1,0,0,0,0,0],[0,1..],...] 
		# is stored but the [2,7,6,9,1,...]
		label = []
		for i in range(len(sublabel)):
			where = np.where(sublabel[i]==1)[0][0]
			label.append(where)
		label = np.array(label).astype(np.float)

		# calc the hist over the label array and add it t the list 
		hist[n] = np.histogram(label, bins = 10)[0]

	return hist

for neuron in range(len(neuron_number_)):
	print "Neuron:" , neuron
	neuron_number = neuron_number_[neuron]


	index_neuron = np.where(DBM.layer_save_test[layer][-1,:,neuron_number]>fire_thresh)[0]
	if len(index_neuron)<4:
		continue


	subdata = test_data[index_neuron]
	sublabel = test_label[index_neuron]


	pca.fit(subdata)
	trans = pca.transform(subdata)
	# kmeans.fit(subdata)
	

	label = []
	for i in range(len(sublabel)):
		where = np.where(sublabel[i]==1)[0][0]
		label.append(where)
	label = np.array(label).astype(np.float)


	fig,ax = plt.subplots(3,2,figsize=(7,9))
	fig.suptitle("Layer %i \n Neuron with Mean %f"%(layer,neuron_means_[neuron]))
	k = 0
	for i in range(2):
		for j in range(2):
			mapp = ax[i,j].matshow(pca.components_[k].reshape(28,28))
			# mapp = ax[i,j].matshow(kmeans.cluster_centers_[k].reshape(28,28))
			plt.colorbar(ax=ax[i,j], mappable=mapp)
			ax[i,j].set_title("Expl. variance = "+str(round(pca.explained_variance_ratio_[k],3)))
			ax[i,j].set_xticks([])
			ax[i,j].set_yticks([])
			k+=1

	mapp = ax[2,0].scatter(trans[::2,0], trans[::2,1], c = label[::2],cmap=plt.cm.get_cmap('spectral', 10),alpha=0.5,vmin=0,vmax=9)
	ax[2,0].set_title("PCA of images with %s firerate"%str(round(fire_thresh,2)))
	plt.colorbar(ax = ax[2,0], mappable = mapp)

	ax[2,1].hist(label,bins=10,align="mid")
	ax[2,1].set_xticks(range(10))
	# plt.savefig(str(neuron),dpi=250)
	# mapp = ax[1,1].matshow(pca.components_[2].reshape(28,28))
	# plt.colorbar(ax=ax[1,1], mappable=mapp)

	# plt.tight_layout()
plt.show()
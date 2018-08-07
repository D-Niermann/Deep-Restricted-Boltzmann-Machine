# from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

kmeans = KMeans(n_clusters=1)
pca = PCA(2)
mean_img = np.mean(train_data[:10000],0)

## user variables
layer       = 3
fire_thresh = 0.9
min_mean    = 0.05
max_mean    = 0.4


# os.chdir("/Users/Niermann/Desktop/Plots/Layer_%i"%layer)
means = [None]*DBM.n_layers
for l in range(DBM.n_layers):
	means_test[l] = np.mean(DBM.firerate_test[l],0)
	means_nc[l]   = np.mean(DBM.firerate_c[l],0)
	means_c[l]    = np.mean(DBM.firerate_nc[l],0)

fig,ax=plt.subplots(3,1,sharex="col")
for l in range(1,4):
	ax[l-1].hist(means[l],bins=30,alpha=0.3)
	ax[l-1].set_title("Average Fire Rate Layer %i"%l)

# neuron_number_ = []
# neuron_means_ =[]
# for i in range(DBM.SHAPE[layer]):
# 	m = means[layer][i]
# 	if m>min_mean and m<max_mean:
# 		neuron_number_.append(np.where(means[layer]==m)[0][0])
# 		neuron_means_.append(m)
# 	if len(neuron_number_)>20:
# 		break

# neuron_number_ = list(set(neuron_number_))
# DBM.n_layers
neuron_number_ = np.where(np.abs(DBM.w_np[-1])>0.04)[0]
target_class = np.where(np.abs(DBM.w_np[-1])>0.04)[1]
# neuron_number_ = [3]


for neuron in range(len(neuron_number_)):
	
	neuron_number = neuron_number_[neuron]
	log.out("Neuron:" , neuron_number)

	index_neuron = np.where(DBM.firerate_test[layer][:,neuron_number]>fire_thresh)[0]
	if len(index_neuron)<4:
		continue


	subdata     = test_data[index_neuron]
	sublabel    = test_label[index_neuron]

	subdata_c   = test_data[index_for_number_gibbs[:]]
	sublabel_c  = test_label[index_for_number_gibbs[:]]

	subdata_nc  = test_data[index_neuroindex_for_number_gibbs[:]n]
	sublabel_nc = test_label[index_neuron]

	mean_subdata = np.mean(subdata,0)

	pca.fit(subdata)
	trans = pca.transform(subdata)
	# kmeans.fit(subdata)
	

	label = []
	for i in range(len(sublabel)):
		where = np.where(sublabel[i]==1)[0][0]
		label.append(where)
	label = np.array(label).astype(np.float)


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
		ax[1].set_title("")

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
	ax[3].hist(label,bins=10,align="mid",lw=0.2,edgecolor="k")
	ax[3].set_xticks(range(10))
	ax[3].set_ylabel("N")
	ax[3].set_xlabel("Class")

	aa[4].set_title("Weight strength")
	ax[4].bar(range(10),DBM.w_np[-1][neuron_number,:])
	ax[4].set_xticks(range(10))
	# plt.savefig(str(neuron),dpi=250)
	# mapp = ax[1,1].matshow(pca.components_[2].reshape(28,28))
	# plt.colorbar(ax=ax[1,1], mappable=mapp)

	plt.subplots_adjust(top=0.65, bottom = 0.17, wspace=0.4, left=0.04, right=0.97)
	            # wspace=None, hspace=None)


log.reset()
plt.show()
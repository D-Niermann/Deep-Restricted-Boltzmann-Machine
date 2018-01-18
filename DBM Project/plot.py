#### Plot
# Plot the Weights, Errors and other informations
if plotting:
	
	map1=plt.matshow(tile_raster_images(X=DBM.w1_np.T, img_shape=(28, 28), tile_shape=(12, 12), tile_spacing=(0,0)))
	plt.title("W 1")


	# map2=plt.matshow(tile_raster_images(X=DBM.w2_np.T, img_shape=(int(sqrt(n_second_layer)),int(sqrt(n_second_layer))), tile_shape=(12, 12), tile_spacing=(0,0)))
	# plt.title("W 2")
	# plt.colorbar(map2)

	if training:
		fig_fr=plt.figure(figsize=(7,9))
		
		ax_fr1=fig_fr.add_subplot(311)
		ax_fr1.plot(DBM.h1_activity_np)
		
		ax_fr2=fig_fr.add_subplot(312)
		# ax_fr2.plot(DBM.CD1_mean_np,label="CD1")
		# ax_fr2.plot(DBM.CD2_mean_np,label="CD2")
		ax_fr2.plot(DBM.w1_mean_np,label="Weights")
		ax_fr1.set_title("Firerate h1 layer")
		ax_fr2.set_title("Weights, CD1 and CD2 mean")
		ax_fr2.legend(loc="best")
		
		ax_fr3=fig_fr.add_subplot(313)
		ax_fr3.plot(DBM.train_error_np,"k")
		ax_fr3.set_title("Train Error")
		
		plt.tight_layout()


	#plot some samples from the testdata 
	fig3,ax3 = plt.subplots(6,15,figsize=(16,4))
	for i in range(15):
		# plot the input
		ax3[0][i].matshow(test_data[i:i+1].reshape(28,28))
		# plot the probs of visible layer
		ax3[1][i].matshow(DBM.probs[i:i+1].reshape(28,28))
		# plot the recunstructed image
		ax3[2][i].matshow(DBM.rec[i:i+1].reshape(28,28))
		# plot the hidden layer h2 and h1
		ax3[3][i].matshow(DBM.h1_test[i:i+1].reshape(int(sqrt(DBM.shape[1])),int(sqrt(DBM.shape[1]))))
		ax3[4][i].matshow(DBM.h2_test[i:i+1,:9].reshape(int(sqrt(DBM.shape[2])),int(sqrt(DBM.shape[2]))))
		#plot the reconstructed layer h1
		ax3[5][i].matshow(DBM.rec_h1[i:i+1].reshape(int(sqrt(DBM.shape[1])),int(sqrt(DBM.shape[1]))))
		# plt.matshow(random_recon.reshape(28,28))
	plt.tight_layout(pad=0.0)

	#plot only one digit
	fig4,ax4 = plt.subplots(6,10,figsize=(16,4))
	m=0
	for i in index_for_number[0:10]:
		# plot the input
		ax4[0][m].matshow(test_data[i:i+1].reshape(28,28))
		# plot the probs of visible layer
		ax4[1][m].matshow(DBM.probs[i:i+1].reshape(28,28))
		# plot the recunstructed image
		ax4[2][m].matshow(DBM.rec[i:i+1].reshape(28,28))
		# plot the hidden layer h2 and h1
		ax4[3][m].matshow(DBM.h1_test[i:i+1].reshape(int(sqrt(DBM.shape[1])),int(sqrt(DBM.shape[1]))))
		ax4[4][m].matshow(DBM.h2_test[i:i+1,:9].reshape(int(sqrt(DBM.shape[2])),int(sqrt(DBM.shape[2]))))
		#plot the reconstructed layer h1
		ax4[5][m].matshow(DBM.rec_h1[i:i+1].reshape(int(sqrt(DBM.shape[1])),int(sqrt(DBM.shape[1]))))
		# plt.matshow(random_recon.reshape(28,28))
		m+=1
	plt.tight_layout(pad=0.0)

	# plot the reverse_feed:
	fig5,ax5 = plt.subplots(2,14,figsize=(16,4))
	for i in range(14):
		ax5[0][i].matshow(test_label[i,:9].reshape(3,3))
		ax5[1][i].matshow(v_rev[i].reshape(28,28))
	plt.tight_layout(pad=0.0)


	plt.matshow(np.mean(v_rev[index_for_number[:]],0).reshape(28,28))

plt.show()
### Propagation (old)
		## Forward Feed
		# h1 gets both inputs from h2 and v
		# self.h1_prob = sigmoid(tf.matmul(self.v,self.w1) + tf.matmul(self.h2_var,self.w2,transpose_b=True) + self.bias_h1,self.temp)
		# self.h1      = self.sample(self.h1_prob)
		
		# # h2 only from h1
		# if graph_mode == "testing" or graph_mode == "gibbs":
		# 	self.h2_prob = sigmoid(tf.matmul(self.h1,self.w2) + self.bias_h2,self.temp)
		# 	self.h2      = self.sample(self.h2_prob)

		# ## Backward Feed   
		# self.h1_recon_prob = sigmoid(tf.matmul(self.v_var,self.w1)+tf.matmul(self.h2_var,self.w2,transpose_b=True)+self.bias_h1, self.temp)
		# self.h1_recon      = self.sample(self.h1_recon_prob)
		# self.v_recon_prob  = sigmoid(tf.matmul(self.h1_recon,self.w1,transpose_b=True)+self.bias_v, self.temp)
		# self.v_recon       = self.sample(self.v_recon_prob)

		# ## Gibbs step 
		# self.h1_gibbs_prob = sigmoid(tf.matmul(self.v_recon_prob,self.w1) + tf.matmul(self.h2,self.w2,transpose_b=True) + self.bias_h1,self.temp)
		# self.h1_gibbs      = self.sample(self.h1_gibbs_prob)
		# self.h2_gibbs_prob = sigmoid(tf.matmul(self.h1_recon_prob,self.w2), self.temp)
		# self.h2_gibbs      = self.sample(self.h2_gibbs_prob)

		### reverse feed (old)
		# self.h2_rev      = tf.placeholder(tf.float32,[None,10],name="reverse_h2")
		# self.h1_rev_prob = sigmoid(tf.matmul(self.v, self.w1) + tf.matmul(self.h2_rev, (self.w2),transpose_b=True)+self.bias_h1,self.temp)
		# self.h1_rev      = tf.nn.relu(tf.sign(self.h1_rev_prob - tf.random_uniform(tf.shape(self.h1_rev_prob)))) 
		# self.v_rev_prob  = sigmoid(tf.matmul(self.h1_rev, (self.w1),transpose_b=True)+self.bias_v,self.temp)
		# self.v_rev       = tf.nn.relu(tf.sign(self.v_rev_prob - tf.random_uniform(tf.shape(self.v_rev_prob)))) 

		#test sample (old)
		# self.h1_place  = tf.placeholder(tf.float32,[None,self.shape[1]],name="h1_placeholder")
		# self.h2_sample = sigmoid(tf.matmul(self.h1_place,self.w2) + self.bias_h2, self.temp)
		# self.v_sample  = sigmoid(tf.matmul(self.h1_place,self.w1,transpose_b=True) + self.bias_v, self.temp)

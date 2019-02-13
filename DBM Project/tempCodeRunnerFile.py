n_visible = UserSettings["DBM_SHAPE"][0]
n_hidden  = UserSettings["DBM_SHAPE"][1]
# define a weight matrix
w = rnd.randn(n_visible, n_hidden)*0.5
# set small elemts to zero
w[np.abs(w)<0.45] = 0
fig = plt.figure("Org Weights")
plt.matshow(w,fig.number)
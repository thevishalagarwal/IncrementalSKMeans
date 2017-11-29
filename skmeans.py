import cPickle
import gzip
import numpy as np
from matplotlib import pyplot as plt
import skmeans_utils as skutil
import scipy.misc
from tqdm import tqdm

def normalize(x):
	return x / np.sqrt(np.sum(x**2, axis=0, keepdims=True))

def showImage(img):
    img = img.reshape((28,28))
    plt.imshow(img, cmap='gray')
    #plt.show()

def angleBetweenVectors(x1, x2):
	a = np.dot(x1.T,x2)
	t = a>=1 
	t1 = a<1.1
	t = t*t1
	try:
		a[:, t[0,:]] = 1.0
	except:
		if t==True:
			a = 1.0
	return np.arccos(a) #*(180.0/np.pi)

def weightedSum(x1, x2, w1, w2):
	return (w1*x1+w2*x2)/(w1+w2)

def getMembership(x, sigma):
	sigma = sigma + 0.1
	return np.exp(-0.5*((x*1.0/sigma)**2))

def appendVector(x1, x2):
	x1 = np.column_stack((x1, x2))
	return x1

def combineClusters(cluster, i, j):
	cluster['mu'][:, i] = weightedSum(cluster['mu'][:, i], cluster['mu'][:, j], cluster['pi_weight'][:, i], cluster['pi_weight'][:, j])
	cluster['mu_theta'][:, i] = weightedSum(cluster['mu_theta'][:, i], cluster['mu_theta'][:, j], cluster['pi_weight'][:, i], cluster['pi_weight'][:, j])
	cluster['sigma_theta'][:, i] = weightedSum(cluster['sigma_theta'][:, i], cluster['sigma_theta'][:, j], cluster['pi_weight'][:, i], cluster['pi_weight'][:, j])
	cluster['pi_weight'][:, i] += cluster['pi_weight'][:, j]

	cluster['mu'] = np.delete(cluster['mu'], j, axis=1)
	cluster['mu_theta'] = np.delete(cluster['mu_theta'], j, axis=1)
	cluster['sigma_theta'] = np.delete(cluster['sigma_theta'], j, axis=1)
	cluster['pi_weight'] = np.delete(cluster['pi_weight'], j, axis=1)

	cluster['mu'] = normalize(cluster['mu'])

	return cluster



X = skutil.loadData()

m = X.shape[1]
n_dim = X.shape[0]
num_cluster = 1
cluster_count = []

#hyperparameters
k = 3
beta = 0.45
gamma = 0.8
merge_threshold = 1
cluster = { 'mu' : np.reshape(X[:, 0], (784,1)),
			'mu_theta' : np.array([[1e-8]]),
			'sigma_theta' : np.array([[0.25]]),
			'pi_weight' : np.array([[1.0]])
}

print '\nClustering data...'

for i in tqdm(range(1000)):
	#print cluster['mu_theta']

	#print 'Running %d iteration'%(i),
	X_i = np.reshape(X[:,i], (784, 1))
	cluster['mu'] = normalize(cluster['mu'])
	theta = angleBetweenVectors(X_i, cluster['mu'])

	host_cluster = theta < abs(cluster['mu_theta'] + k*cluster['sigma_theta'])
	
	# No host cluster => Create new cluster
	if np.all(~host_cluster) == True:
		num_cluster += 1
		cluster['mu'] = appendVector(cluster['mu'], X[:, i])
		cluster['mu_theta'] = appendVector(cluster['mu_theta'], np.array([[1e-8]]))
		cluster['sigma_theta'] = appendVector(cluster['sigma_theta'], np.array([[0.25]]))
		cluster['pi_weight'] = appendVector(cluster['pi_weight'], np.array([[1.0]]))

		cluster['mu'] = normalize(cluster['mu'])

	else:
		# Get Membership Values
		membrship_host = getMembership(theta[:, host_cluster[0,:]], cluster['sigma_theta'][:, host_cluster[0,:]])
		
		alpha = membrship_host/(np.sum(membrship_host))
	
	
		# Update Cluster parameters
		cluster['mu'][:, host_cluster[0,:]] = (1-alpha)* cluster['mu'][:, host_cluster[0,:]] + X_i
		cluster['mu'] = normalize(cluster['mu'])
		#theta = angleBetweenVectors(X_i, cluster['mu'])
		cluster['mu_theta'][:, host_cluster[0,:]] = (1 - beta)*cluster['mu_theta'][:, host_cluster[0,:]] + beta*theta[:, host_cluster[0,:]]
		cluster['sigma_theta'][:, host_cluster[0,:]] = np.sqrt(abs((1-beta)*(cluster['sigma_theta'][:, host_cluster[0,:]]**2 + beta*(theta[:, host_cluster[0,:]] - cluster['mu_theta'][:, host_cluster[0,:]])**2))) 
		cluster['pi_weight'][:, host_cluster[0,:]]  = (1 - gamma*alpha)*cluster['pi_weight'][:, host_cluster[0,:]]  + gamma*alpha

		# Penalize non-host clusters
		cluster['pi_weight'][:,~host_cluster[0,:]] -= 1.0/(i+1)

	
	# #Merge clusters
	# if i%500==0:
	# 	j = 0
	# 	while j<num_cluster:
	# 		k = j+1
	# 		while k<num_cluster:
	# 			phi = angleBetweenVectors(cluster['mu'][:, j], cluster['mu'][:, k])
	# 			if ((phi/2)*(1/cluster['sigma_theta'][0,j] + 1/cluster['sigma_theta'][0,k])) < merge_threshold:
	# 				cluster = combineClusters(cluster, j, k)
	# 				num_cluster -= 1
	# 				break
	# 			k += 1
	# 		j += 1	
	#print num_cluster
	cluster_count.append(num_cluster)

print 'Clustering complete.'
print '\n\tNumber of clusters formed : %d\n'%(num_cluster)
print 'Saving templates...'

# for k in range(num_cluster):
# 	print 'Cluster '+ str(k) + ' : ',
# 	for l in range(10):
# 		#print np.dot(cluster['mu'][:,k].T, cluster['mu'][:, l]),
# 		c = cluster['mu'][:, l]
#     	# plt.figure()
#     	# showImage(c)
#     	print ''

for i in tqdm(range(num_cluster)):
	img = cluster['mu'][:, i]
	img = img.reshape((28,28))
	# plt.imshow(img, cmap='gray')
	# plt.savefig('z'+str(i)+'.png')
	scipy.misc.imsave('z'+str(i)+'.png', img)


# plt.ion() # turn on interactive mode
# fig = plt.figure()
# plt.plot(cluster_count)
# plt.title('Cluster count after each iteration')
# plt.ylabel('Number of clusters')
# plt.xlabel('Number of data')
# plt.grid()
#plt.show()
#fig.savefig('c2.png')

#_ = raw_input("Press [Enter] to close all windows.")
print 'Done!'
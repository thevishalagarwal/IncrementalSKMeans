import cPickle
import gzip
import numpy as np
from matplotlib import pyplot as plt

def normalize(x):
	return x / np.sqrt(np.sum(x**2, axis=0))

def showImage(img):
	img = img.reshape((28,28))
	plt.imshow(img, cmap='gray')
	plt.show()

def angleBetweenVectors(x1, x2):
	a = np.arccos(np.dot(x1.T,x2))
	return a*180/np.pi

def weightedSum(x1, x2, w1, w2):
	return (w1*x1+w2*x2)/(w1+w2)

def getMembership(x, sigma):
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

def loadData():
	f = gzip.open('mnist.pkl.gz', 'rb')
	x, _, _ = cPickle.load(f)
	X = x[0]	
	X = X.T # X.shape = (784, 50000)
	np.random.shuffle(X.T)
	X = normalize(X)
	return X

def newCluster(cluster, X, theta):
	cluster['mu'] = appendVector(cluster['mu'], X)
	cluster['mu_theta'] = appendVector(cluster['mu_theta'], np.mean(theta))
	cluster['sigma_theta'] = appendVector(cluster['sigma_theta'], abs(np.std(theta)))
	cluster['pi_weight'] = appendVector(cluster['pi_weight'], np.array([[1.0]]))

	cluster['mu'] = normalize(cluster['mu'])
	return cluster

def addToCluster(cluster, X, num_cluster, k, beta, gamma):
	mu = cluster['mu']
	muTheta = cluster['mu_theta']
	sigmaTheta = cluster['sigma_theta']
	pi = cluster['pi_weight']
	
	theta = angleBetweenVectors(X, mu)
	
	# Get Host Clusters
	host_cluster = abs(theta) < abs( muTheta + k*sigmaTheta )
	host = host_cluster[0, :]
	
	# No host cluster => Create new cluster
	if np.all(~host_cluster) == True:
		num_cluster += 1
		return newCluster(cluster, X, theta), num_cluster
	
	# Get Membership Values
	membrship_host = getMembership(theta[:, host], sigmaTheta[:, host])
	alpha = membrship_host/(np.sum(membrship_host, keepdims=True))
	
	# Update Cluster parameters
	mu[:, host] = (1-alpha)* mu[:, host] + X
	muTheta[:, host] = (1 - beta)*muTheta[0, host] + beta*theta[:, host]
	sigmaTheta[:, host] = np.sqrt(abs((1-beta)*(sigmaTheta[:, host]**2 + beta*(theta[:, host] - muTheta[:, host])**2))) 
	pi[:, host]  = (1 - gamma*alpha)*pi[:, host]  + gamma*alpha

	# Penalize non-host clusters
	pi[:,~host] -= gamma

	mu = normalize(mu) #normalize mean vector
	cluster['mu'] = mu
	cluster['mu_theta'] = muTheta
	cluster['sigma_theta'] = sigmaTheta
	cluster['pi_weight'] = pi

	return cluster, num_cluster

def mergeCluster(cluster, num_cluster, merge_threshold):

	mu = cluster['mu']
	muTheta = cluster['mu_theta']
	sigmaTheta = cluster['sigma_theta']
	pi = cluster['pi_weight']

	for j in range(num_cluster-1):
		for k in range(j+1, num_cluster-1):
			phi = angleBetweenVectors( mu[:, j], mu[:, k] )
				
			if ( (phi/2)*(1/sigmaTheta[0,j] + 1/sigmaTheta[0,k]) ) < merge_threshold:
				cluster = skutil.combineClusters(cluster, j, k)
				num_cluster -= 1

	mu = normalize(mu)
	cluster['mu'] = mu
	cluster['mu_theta'] = muTheta
	cluster['sigma_theta'] = sigmaTheta
	cluster['pi_weight'] = pi

	return cluster, num_cluster
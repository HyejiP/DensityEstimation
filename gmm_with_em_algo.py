import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn


# load data (we don't need to scale the data as each pixels value is in a fixed range)
mat1 = scipy.io.loadmat('data/data.mat')
mat2 = scipy.io.loadmat('data/label.mat')

data = mat1['data'].T
label = mat2['trueLabel'].reshape(-1,)

m, n = data.shape
C = np.matmul(data.T, data)/m

# pca the data
d = 4  # reduced dimension 
U, V,_ = np.linalg.svd(C)
U = U[:, :d]
V = V[:d]

# project the data to the top 4 principal directions
pdata = np.dot(data,U)

K = 2 # number of Gaussians
np.random.seed(30)

pi = np.random.random(K)
pi = pi/np.sum(pi)

# random initialization of mu 
mu = np.random.randn(K, d)
mu_old = mu.copy()

# random initialization of sigma                      
sigma = []
for kk in range(K):
    dummy = np.random.normal(0, 1, size=(d, d))
    sigma.append(dummy@dummy.T + np.eye(d))
    
# initialize the posterior
tau = np.full((m, K), fill_value=0.)

maxIter= 100
tol = 1e-3 # threshold to determine whether or not the model converged

log_likelihood = [] # later, we will plot log likelihood
for ii in range(100):

    # E-step    
    for kk in range(K):
        tau[:,kk] = pi[kk] * mvn.pdf(pdata, mu[kk], sigma[kk])
    # normalize tau
    sum_tau = np.sum(tau, axis=1)
    sum_tau.shape = (m,1)    
    tau = np.divide(tau, np.tile(sum_tau, (1, K)))
    
    
    # M-step
    for kk in range(K):
        # update prior
        pi[kk] = np.sum(tau[:,kk])/m
        
        # update component mean
        mu[kk] = pdata.T @ tau[:,kk] / np.sum(tau[:,kk], axis = 0)
        
        # update cov matrix
        dummy = pdata - np.tile(mu[kk], (m,1)) 
        sigma[kk] = dummy.T @ np.diag(tau[:,kk]) @ dummy / np.sum(tau[:,kk], axis = 0)
    
    # compute log-likelihood
    tau_1 = []
    tau_2 = []
    for i in range(m):
        if np.argmax(tau[i,:]) == 0:
            tau_1.append(np.log(tau[i, 0]))
        else:
            tau_2.append(np.log(tau[i, 1]))
        
    log_likelihood.append(sum(tau_1) + sum(tau_2))
    

    print('-----iteration---',ii)    
    if np.linalg.norm(mu-mu_old) < tol:
        print('training coverged')
        break
    mu_old = mu.copy()
    if ii==99:
        print('max iteration reached')
        break

# plot log-likelihood
plt.plot(log_likelihood)
plt.title('Log-likelihood', fontweight='bold', color='orange')
plt.show()


print('----Fitted GMM model----')
print('-mu: \n', mu)
print('-sigma: \n', sigma)


# helper function to reconstruct the images
def reconstruct_img(U, V, mu):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    fig.suptitle('Average Images', fontweight='bold', color='orange')
    for i in range(2):
        axs[i].imshow((np.dot(np.dot(U, np.sqrt(np.diag(V))), mu[i])).reshape(28, 28))
        axs[i].axis('off')
    return fig, axs

fig, axs = reconstruct_img(U=U, V=V, mu=mu)
plt.show()


# visualize covariance matrices
fig, axs = plt.subplots(1, 2, figsize=(6, 3))
fig.suptitle('Visualization Of Covariance Matrices', fontweight='bold', color='orange')
for i in range(2):
    axs[i].matshow(sigma[i])
    axs[i].axis('off')
plt.show()


# predict labels based on the converged tau
labels = [2, 6] # the order of labels(2 and 6, or, 6 and 2) should be decided upon observing the average images
def get_pred_label(tau, m, labels):
    pred_label = []
    for i in range(m):
        if np.argmax(tau[i, :]) == 0:
            pred_label.append(labels[0])
        else:
            pred_label.append(labels[1])
    return np.array(pred_label)

pred_label = get_pred_label(tau=tau, m=m, labels=labels)


# compute misclassification rate of each label respectively
mis_rates = []
for l in labels:
    mis = sum(pred_label[np.where(label == l)] != l) / sum(label == l)
    mis_rates.append(round(mis, 4))

print('**below are the misclassification rates of GMM-EM algorithm')
print('-misclassification rate of label {} is {}%'.format(labels[0], mis_rates[0] * 100))
print('-misclassification rate of label {} is {}%'.format(labels[1], mis_rates[1] * 100))


###########################################################################################
# conduct k-mean algorithm and get misclassification rate
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2, random_state=30)
km.fit(pdata)

km_label = np.empty(m)
km_label[np.where(km.labels_ == 1)] = 2
km_label[np.where(km.labels_ == 0)] = 6

km_mis_rates = []
for l in labels:
    mis = sum(km_label[np.where(label == l)] != l) / sum(label == l)
    km_mis_rates.append(round(mis, 4))

print('**below are the misclassification rates of K-Means algorithm')
print('-misclassification rate of label {} is {}%'.format(labels[0], km_mis_rates[0] * 100))
print('-misclassification rate of label {} is {}%'.format(labels[1], km_mis_rates[1] * 100))

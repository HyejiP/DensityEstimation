import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.stats import gaussian_kde

# Read the data in first 
data = pd.read_csv('data/n90pol.csv').to_numpy()

# Preprocess the data
label = data[:,2].astype(int) # the last column of the original data which is a set of labels of political views
amyg = preprocessing.scale(data[:,0]) # the column of amydala data
acc = preprocessing.scale(data[:,1]) # the column of acc data


################# Below is code for 1-dimensional histograms ####################################
# Plot histograms of amygdala with different numbers of bins 
fig, axs = plt.subplots(2, 5, figsize=(20, 4))
fig.suptitle('1D Histograms Of Amygdala', fontweight='bold', color='orange')
for i in range(2):
    for j in range(5):
        axs[i, j].hist(amyg, bins=(5+i*5+j), histtype='stepfilled', linewidth=1) # trying different # of bins(5~14 bins)
        axs[i, j].set_title(str(5+i*5+j) + ' bins', y=-0.01)
plt.show()

# Find the best number of bins using Freedman-Diaconis rule
q3 = np.quantile(amyg, 0.75)
q1 = np.quantile(amyg, 0.25)
bin_width = (2 * (q3 - q1)) / (len(amyg) ** (1 / 3))
bin_count = int(np.ceil((amyg.max() - amyg.min()) / bin_width))
print('-best number of bins for amygdala data: ', bin_count)

# Print the data counts in each bean and edges of each bin
amyg_hist, amyg_bins = np.histogram(amyg, bins=bin_count)
print('-data counts in each bin: ', amyg_hist)
print('-edges of bins: ', amyg_bins)


# Plot histograms of acc with different numbers of bins 
fig, axs = plt.subplots(2, 5, figsize=(20, 4))
fig.suptitle('1D Histograms Of ACC', fontweight='bold', color='orange')
for i in range(2):
    for j in range(5):
        axs[i, j].hist(acc, bins=(5+i*5+j), histtype='stepfilled', linewidth=1)
        axs[i, j].set_title(str(5+i*5+j) + ' bins', y=-0.01)
plt.show()

# Find the best number of bins using Freedman-Diaconis rule
q3 = np.quantile(acc, 0.75)
q1 = np.quantile(acc, 0.25)
bin_width = (2 * (q3 - q1)) / (len(acc) ** (1 / 3))
bin_count = int(np.ceil((acc.max() - acc.min()) / bin_width))
print('-best number of bins for acc data: ', bin_count)

# Print the best number of bins, the data counts in each bin, and edges of each bin
acc_hist, acc_bins = np.histogram(acc, bins=bin_count)
print('-data counts in each bin: ', acc_hist)
print('-edges of bins: ', acc_bins)




################# Below is code for 1-dimensional KDE ####################################
# Plot KDE's of amygdala with different bandwidths
bandwidth = np.arange(0.16, 0.36, 0.02)

# Construct Gaussian KDE's with amygdala data and plot them
x = amyg
density = gaussian_kde(x)
x_vals = np.linspace(-4, 4, 200) 

fig, axs = plt.subplots(2, 5, figsize=(20, 5))
fig.suptitle("1D KDE's Of Amygdala", fontweight='bold', color='orange')
for i in range(2):
    for j in range(5):
        density.covariance_factor = lambda : bandwidth[i*5 + j]
        density._compute_covariance()
        axs[i, j].plot(x_vals, density(x_vals))
        axs[i, j].set_title('h = '+ str(round(bandwidth[i*5+j], 2)), y=-0.01)
plt.show()

# Print the best bandwidth for KDE for amygdala data
params = {'bandwidth': np.arange(0.16, 0.36, 0.02)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit(x.reshape(-1, 1))
print('-best bandwidth for amygdala data: {0}'.format(grid.best_estimator_.bandwidth))


# Construct Gaussian KDE's with acc data and plot them
x = acc
density = gaussian_kde(x)
x_vals = np.linspace(-4, 4, 200)

fig, axs = plt.subplots(2, 5, figsize=(20, 5))
fig.suptitle("1D KDE's Of Acc", fontweight='bold', color='orange')
for i in range(2):
    for j in range(5):
        density.covariance_factor = lambda : bandwidth[i*5 + j]
        density._compute_covariance()
        axs[i, j].plot(x_vals, density(x_vals))
        axs[i, j].set_title('h = '+ str(round(bandwidth[i*5+j], 2)), y=-0.01)
plt.show()

# Print the best bandwidth for KDE for acc data
params = {'bandwidth': np.arange(0.16, 0.36, 0.02)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit(x.reshape(-1, 1))
print('-best bandwidth acc data: {0}'.format(grid.best_estimator_.bandwidth))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import gaussian_kde

# Read the data in first 
data = pd.read_csv('data/n90pol.csv').to_numpy()

# Preprocess the data
label = data[:,2].astype(int) # the last column of the original data which is a set of labels of political views
amyg = preprocessing.scale(data[:,0]) # the column of amydala data
acc = preprocessing.scale(data[:,1]) # the column of acc data


################# Below is code for plotting conditional distributions ####################################
orientation = [2, 3, 4, 5] # a list of political orientation

# Plot the conditional distributions of amygdala data
fig, axs = plt.subplots(1, 4, figsize=(24, 3))
fig.suptitle('Conditional Distributions Of Amygdala Data', fontweight='bold', color='orange')
for i, c in enumerate(orientation):
    x = amyg[label==c]
    density = gaussian_kde(x)
    x_vals = np.linspace(-4, 4, 200) 

    density.covariance_factor = lambda : 0.18
    density._compute_covariance()
    axs[i].plot(x_vals, density(x_vals))
    axs[i].set_title('political orientation = ' + str(c), y=0.01)
plt.show()

# Plot the conditional distributions of acc data
fig, axs = plt.subplots(1, 4, figsize=(24, 3))
fig.suptitle('Conditional Distributions Of ACC Data', fontweight='bold', color='orange')
for i, c in enumerate(orientation):
    x = acc[label==c]
    density = gaussian_kde(x)
    x_vals = np.linspace(-4, 4, 200) 

    density.covariance_factor = lambda : 0.18
    density._compute_covariance()
    axs[i].plot(x_vals, density(x_vals))
    axs[i].set_title('political orientation = ' + str(c), y=0.01)
plt.show()

# Overlay conditional distributions of amygdala data
for i, c in enumerate(orientation):
    x = amyg[label==c]
    density = gaussian_kde(x)
    x_vals = np.linspace(-4, 4, 200) 

    density.covariance_factor = lambda : 0.18
    density._compute_covariance()
    plt.plot(x_vals, density(x_vals), alpha=0.5)
    plt.title('Overlaid Conditional Distributions Of Amygdala Data', fontweight='bold', color='orange')
    plt.legend(labels=['orientation=2', 'orientation=3', 'orientation=4', 'orientation=5'])
plt.show()

# Overlay conditional distributions of acc data
for i, c in enumerate(orientation):
    x = acc[label==c]
    density = gaussian_kde(x)
    x_vals = np.linspace(-4, 4, 200) 

    density.covariance_factor = lambda : 0.18
    density._compute_covariance()
    plt.plot(x_vals, density(x_vals), alpha=0.5)
    plt.title('Overlaid Conditional Distributions Of ACC Data', fontweight='bold', color='orange')
    plt.legend(labels=['orientation=2', 'orientation=3', 'orientation=4', 'orientation=5'])
plt.show()



################# Below is code for computing conditional mean ####################################
data = pd.read_csv('data/n90pol.csv').to_numpy()
cond_mean_amygdala = []
cond_mean_acc = []
for c in orientation:
    cond_mean_amygdala.append(round(np.mean(data[:,0][data[:,2] == c]), 4))
    cond_mean_acc.append(round(np.mean(data[:,1][data[:,2] == c]), 4))

print('-conditional means of amygdala data: \n', cond_mean_amygdala)
print('-conditional means of acc data: \n', cond_mean_acc)
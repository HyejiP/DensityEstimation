import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import gaussian_kde

# Read the data in first
data = pd.read_csv('data/n90pol.csv').to_numpy()

# Preprocess the data
label = data[:,2].astype(int)
amyg = preprocessing.scale(data[:,0])
acc = preprocessing.scale(data[:,1])

# Read the data in first
data = pd.read_csv('data/n90pol.csv').to_numpy()

# Preprocess the data
label = data[:,2].astype(int)
amyg = preprocessing.scale(data[:,0])
acc = preprocessing.scale(data[:,1])

xmin = amyg.min()
xmax = amyg.max()
ymin = acc.min()
ymax = acc.max()

orientation = [2, 3, 4, 5]

fig, axs = plt.subplots(1, 4, figsize=(24, 3))
fig.suptitle('Joint Conditional Distribution', fontweight='bold', color='orange')
for i, c in enumerate(orientation):
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([amyg[label==c], acc[label==c]])
    kernel = gaussian_kde(values, bw_method='silverman')
    Z = np.reshape(kernel(positions).T, X.shape)

    axs[i].imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    axs[i].plot(amyg[label==c], acc[label==c], 'k.', markersize=2)
    axs[i].set_xlim([xmin, xmax])
    axs[i].set_ylim([ymin, ymax])
    axs[i].set_title('orientation = ' + str(c), y=0.01)
plt.show()
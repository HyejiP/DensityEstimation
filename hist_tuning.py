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

# Plot 2D contour plot using both amygdala and acc data
# Tune the bandwidth using silverman's rule
xmin = amyg.min()
xmax = amyg.max()
ymin = acc.min()
ymax = acc.max()

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([amyg, acc])
kernel = gaussian_kde(values, bw_method='silverman')
Z = np.reshape(kernel(positions).T, X.shape)

fig, ax = plt.subplots()
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
          extent=[xmin, xmax, ymin, ymax])
ax.plot(amyg, acc, 'k.', markersize=2)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
plt.title('2D Contour Plot Of KDE', fontweight='bold', color='orange')
plt.show()




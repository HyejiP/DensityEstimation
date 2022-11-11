import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Read the data in first
data = pd.read_csv('data/n90pol.csv').to_numpy()

# Preprocess the data
label = data[:,2].astype(int)
amyg = preprocessing.scale(data[:,0])
acc = preprocessing.scale(data[:,1])


################# Below is code for 2-dimensional histogram ####################################
min_data = min(min(amyg), min(acc))
max_data = max(max(amyg), max(acc))
# In the previous question 2_(a), we found the optimal number of bins for each amygadala data and acc data
# It was n=8 for amygadala and n=9 for acc. I will choose the bigger number so that I don't lose information from both data.
nbin = 9     
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
hist, xedges, yedges = np.histogram2d(amyg, acc, bins=nbin)
xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])
xpos = xpos.flatten()/2.
ypos = ypos.flatten()/2.
zpos = np.zeros_like (xpos)
dx = xedges [1] - xedges [0]
dy = yedges [1] - yedges [0]
dz = hist.flatten()
ax.bar3d(xpos, ypos, zpos, dx, dy, dz )
plt.title('2D Histogram \nUsing Both Amygdala and ACC Data', fontweight='bold', color='orange')
plt.show()
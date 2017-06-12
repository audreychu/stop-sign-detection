import pickle
import numpy as np



with open("heatmaparray.pickle",'rb') as mat:
    lister = pickle.load(mat)


lister = np.array(lister)
lister = lister.reshape((21,31))
print(lister)

from matplotlib import pyplot
import matplotlib as mpl
# make a color map of fixed colors

#bounds=[-6,-2,2,6]
#norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# tell imshow about color map so that only set colors are used
img = pyplot.imshow(lister,cmap = 'coolwarm')

# make a color bar
pyplot.colorbar(img,cmap='coolwarm')

pyplot.show()
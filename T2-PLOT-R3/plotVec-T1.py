import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



p0 = [1,0,0]
p1 = [0,1,0]
p2 = [0, 0, 1]
p3 = [1, 1, 1]





origin = [0,0,0]
X, Y, Z = zip(origin,origin,origin)
U, V, W = zip(p0,p1,p2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X,Y,Z,U,V,W,length=0.1, normalize=True)
plt.show()

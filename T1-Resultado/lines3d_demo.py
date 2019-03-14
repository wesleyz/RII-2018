import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10


p0 = [1,0,0]
p1 = [0,1,0]
p2 = [0, 0, 1]
p4 = [1, 1, 1]



fig = plt.figure()
ax = fig.gca(projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)

x,y,z = zip(p0, p1, p2,p4)



ax.plot(x, y, z, label='parametric curve')
ax.legend()

plt.show()

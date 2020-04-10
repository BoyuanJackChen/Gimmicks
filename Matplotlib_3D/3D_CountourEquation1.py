import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

# Equation: x+y+z-1 = c

c = 4
def f1(x, y):
    return 1-x-y+c

def f2(x, y): 
    return 1-x-y-c

def fplane(x, y): 
    return 1-x-y

x = np.linspace(-6, 6, num=100)
y = np.linspace(-6, 6, num=100)

X, Y = np.meshgrid(x, y) # To [array[1,2,3]]
Z = f1(X, Y)
W = f2(X, Y)
P = fplane(X, Y)
print("X is: ")
print(X.shape)
print("Y is: ")
print(Y.shape)
print("Z is: ")
print(Z.shape)

plt.figure(1)
ax = plt.axes(projection='3d')
#ax.contour3D(X, Y, Z, 100, cmap='binary')
#ax.contour3D(X, Y, W, 100, cmap='binary')
ax.contour3D(X, Y, P, 100, cmap='plasma')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(20, 35)
plt.legend(['Upper'],loc=0)

plt.figure("Wire")
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)
ax.plot_wireframe(X, Y, W, rstride=2, cstride=2)
ax.plot_wireframe(X, Y, P, rstride=2, cstride=2, color='r')  # Try put "100" after P
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(20, 35)

plt.show()
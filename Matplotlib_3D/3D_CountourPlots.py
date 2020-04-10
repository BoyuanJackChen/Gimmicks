import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(-6, 6, num=100)
y = np.linspace(-6, 6, num=100)

X, Y = np.meshgrid(x, y) # To [array[1,2,3]]
Z = f(X, Y)
print("X is: ")
print(X.shape)

print("Y is: ")
print(Y.shape)

print("Z is: ")
print(Z.shape)


plt.figure(1)
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 100, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(20, 35)

plt.figure(2)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=10,
                cmap='plasma', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(20, 35)

plt.show()
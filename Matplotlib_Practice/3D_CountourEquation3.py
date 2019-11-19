import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

# Equation: x^2 + y^2 - z^2 = 1

c = 100
def f1(x, y):
    return np.sqrt(x**2 + y**2 - 1)

def f2(x, y): 
    return -np.sqrt(x**2 + y**2 - 1)

x = np.linspace(-6, 6, num=100)
y = np.linspace(-6, 6, num=100)

X, Y = np.meshgrid(x, y) # To [array[1,2,3]]
Z = f1(X, Y)
W = f2(X, Y)
print("x is: ", x)
print("X is: ", X)
print("Y is: ")
print(Y.shape)
print("Z is: ")
print(Z.shape)

plt.figure(1)
ax = plt.axes(projection='3d')
ax.contour3D(x, y, Z, 100, cmap='plasma')
ax.contour3D(X, Y, W, 100, cmap='plasma')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(20, 35)

plt.figure(2)


plt.show()
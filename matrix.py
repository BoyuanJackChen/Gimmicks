import numpy as np
import matplotlib.pyplot as plt

# --- Solve Ax=y ---
dots = np.array([[-2,3], [-1,1], [0,1], [1,2], [2,4]])
X = np.array([-2,-1,0,1,2])
A = np.matrix([[4, -2, 1],
              [1, -1, 1],
              [0, 0,  1],
              [1, 1,  1],
              [4, 2,  1]])
y = np.matrix([3,1,1,2,4]).reshape(-1,1)
A_star = A.getH()
x = np.linalg.solve(A_star * A, A_star * y)
print(x)

A = np.matrix([[0, 0, 1, 1],
               [0, 0, 0, 1],
               [0, 1, 0, 0],
               [0, 0, 0, 0]])
print(A*A)
print(np.poly(A))


# Plot x and y
# x_f = np.linspace(-2.5, 2.5, 1000)
# y_f = 0.643 * np.square(x_f) + 0.3 * x_f + 0.914
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# print(X.shape)
# y = np.array(y.flat)
# ax.scatter(X, y)
# ax.scatter(x_f, y_f)
# plt.show()
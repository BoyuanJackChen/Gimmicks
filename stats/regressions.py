import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.interpolate import *
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# --- Data ---
# Year
x = np.arange(1801.0, 1941.0, 10)
x_smooth = np.arange(1801.0, 1941.0, 1)
x_2d = x.reshape(-1, 1)
# Population Size
y = np.array([8.89, 10.16, 12.00, 13.90, 15.91, 17.93, 20.07, 22.71,
              25.97, 29.00, 32.53, 36.07, 37.89, 39.95])
y_2d = y.reshape(-1,1)
X = np.column_stack((x,y))


# --- Linear Regression ---
model = LinearRegression()
model.fit(x_2d, y_2d, 1)
curve = np.polyfit(x, y, 1)
print(curve)
poly = np.poly1d(curve)
print(poly)
y_pred = model.predict(x_2d)
print(r2_score(y_2d, y_pred))

fig = plt.figure(1)
ax = plt.axes()
ax.scatter(x, y)
plt.title('Population of England and Wales')
plt.xlabel('Year')
plt.ylabel('Population Size')

ax.plot(x, y_pred, color='green', linewidth=2)


# --- Exponential Regression ---
y_log = np.log(y)
print(y_log)
curve = np.polyfit(x, y_log, 1)
poly = np.poly1d(curve)
print(poly)
print(curve)
print()

y_pred = x
# print(x)   这时候x还是对的
for i in range(0, len(y_pred)):
    y_pred[i] = np.exp(curve[1]+curve[0]*x[i])
print(y_pred[0])

y_pred_smooth = x_smooth
for i in range(0, len(y_pred_smooth)):
    y_pred_smooth[i] = np.exp(curve[1])*np.exp(curve[0]*x_smooth[i])

x_smooth = np.arange(1801, 1941, 1)
fig = plt.figure(2)
ax = plt.axes()
x = np.arange(1801.0, 1941.0, 10)
ax.scatter(x, y)
ax.plot(x_smooth, y_pred_smooth, color='red', linewidth=2)
y_pred_2d = y_pred.reshape(-1,1)
print(r2_score(y_2d, y_pred_2d))



# --- Power Regression ---
curve = np.polyfit(np.log(x), np.log(y), 1)
poly = np.poly1d(curve)
print("Parameters for power are: ")
print(curve)
print(poly)
print("--- Power ---")

x = np.arange(1801.0, 1941.0, 10)
x_smooth = np.arange(1801.0, 1941.0, 1)
y_pred = x
# print(x)   这时候x还是对的
for i in range(0, len(y_pred)):
    y_pred[i] = np.exp(curve[1])*(x[i]**curve[0])
print(y_pred[0])

y_pred_smooth = x_smooth
for i in range(0, len(y_pred_smooth)):
    y_pred_smooth[i] = np.exp(curve[1])*(x_smooth[i]**curve[0])

x_smooth = np.arange(1801, 1941, 1)
fig = plt.figure(3)
ax = plt.axes()
x = np.arange(1801.0, 1941.0, 10)
ax.scatter(x, y)
ax.plot(x_smooth, y_pred_smooth, color='red', linewidth=2)
y_pred_2d = y_pred.reshape(-1,1)
print(r2_score(y_2d, y_pred_2d))
print(y_2d[0], y_pred_2d[0])
plt.show()
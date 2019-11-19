import matplotlib.pyplot as plt

x1 = [1,3,5]
y1 = [1,4,3]
x2 = [2,4,6]
y2 = [4,7,6]

plt.figure(1)
plt.bar(x1,y1,label="Bar_1", color='r')
plt.bar(x2,y2,label="Bar_2", color='b')
plt.legend()

plt.figure(2)
ages = [1,18,27,5,3,19,16,8,10,9,21,22]
bins = [0,10,20,30]
plt.hist(ages, bins, histtype='bar', label='Age Graph')
plt.legend()

plt.show()
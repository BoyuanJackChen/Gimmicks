import matplotlib.pyplot as plt

x1 = [1,2,3]
y1 = [1,4,3]
y2 = [4,7,6]
plt.plot(x1,y1,label="First Line")
plt.plot(x1,y2,label="Second Line")

plt.xlabel("Aha!")
plt.ylabel("Yes!", fontsize=12)
plt.title('Experiment!', fontsize=12)
plt.legend()

plt.show()
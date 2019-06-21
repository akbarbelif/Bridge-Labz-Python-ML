import matplotlib.pyplot as plt
import numpy as np
X=np.linspace(0,2,100)
Y=[value * 1.5 for value in X]
print("Value of X :\n",*range(1,50))
print("Values of Y is thrice then X",Y)
#Pilot Line 
plt.plot(X,X)
#plt.plot(X,X**2)
#plt.plot(X,X**3)
#set Label for X axis
plt.xlabel("X-axis")
#set Label for Y axis
plt.ylabel("Y-axis")
plt.title("Simple Plot")
plt.legend()
plt.show()
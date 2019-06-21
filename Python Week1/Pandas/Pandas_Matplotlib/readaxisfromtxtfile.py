import matplotlib.pyplot as plt

with open("test.txt") as f:
    data=f.read()
    print(data)
data=data.split("\n")
x = [row.split(' ')[0] for row in data]
y = [col.split(' ')[1] for col in data]
print(x)
print(y)
plt.plot(x,y)
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.legend()
plt.show()
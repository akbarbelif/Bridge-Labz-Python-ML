import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

date = []
opened = []
high = []
low = []
close = []
xname = ''
yname = ''
with open('fdata.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        print(row)
        date.append(row[0])
        opened.append(row[1])
        high.append(row[2])
        low.append(row[3])
        close.append(row[4])
df=pd.DataFrame(opened,date)
print("DataFrame: ",df)
plt.plot(date[1:len(date)],opened[1:len(opened)],label="Open")
plt.plot(date[1:len(date)],high[1:len(high)],label="High")
plt.plot(date[1:len(date)],low[1:len(low)],label="Low") 
plt.plot(date[1:len(date)],close[1:len(close)],label="Close")
#plt.plot(date[1:len(date)],opened[1:len(opened)],label="Date")
plt.xlabel(date[0])
plt.ylabel(opened[0])
plt.title('fdata.csv file values')
plt.legend()
plt.show()
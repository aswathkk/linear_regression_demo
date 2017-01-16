import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

#read data
data = pd.read_csv("challenge_dataset.txt", header = None)
x = data[[0]]
y = data[[1]]

#train model on data
linear_reg = linear_model.LinearRegression()
linear_reg.fit(x, y)

#selecting data point
xp = x[30:50]
yp = y[30:50]

print(xp)

#printing error
print("Mean squared error = %f" % np.mean((yp - linear_reg.predict(xp)) ** 2))

#visualize results
plt.scatter(x, y)
plt.plot(x, linear_reg.predict(x))
plt.show()
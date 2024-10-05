import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

time_studied = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 99]).reshape(-1, 1)
scores = np.array([35, 60, 59, 74, 93, 113, 115, 145, 152, 178]).reshape(-1, 1)

model = LinearRegression()
model.fit(time_studied, scores)

# linspace generates evenly spaced numbers over a specified range
x_range = np.linspace(0, 100, 100).reshape(-1, 1)

plt.scatter(time_studied.reshape(-1, 1), scores.reshape(-1, 1))
plt.plot(x_range, model.predict(x_range), 'r', label='Regression Line')
plt.ylim(0, 175)
plt.show()

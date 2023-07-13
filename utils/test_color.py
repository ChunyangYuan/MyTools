import matplotlib.pyplot as plt
import numpy as np

# define data
x = [1, 2, 3, 4, 5]
y = [7, 12, 15, 19, 25]

# create scatterplot with random colors for each point
plt.scatter(x, y, c=np.random.rand(len(x), 3))
plt.show()

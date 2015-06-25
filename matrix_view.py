# condig: utf-8

import numpy as np
import matplotlib.pyplot as plt


w = 100
h = 100

x = np.random.randint(2, size=w*h)
X = x.reshape(w,h)
print X

plt.imshow(X, interpolation='none', cmap='gray')
plt.show()


# Transforming discrete baryonic mass distributions to EG distributions

#!/usr/bin/python

# Import the necessary libraries
import numpy as np
import os

x = np.array([0, 1, 2, 3, 4])
y = x**2
dydx = np.gradient(y, x, edge_order=2)

print(dydx)

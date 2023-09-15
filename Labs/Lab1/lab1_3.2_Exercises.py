import numpy as np
import matplotlib.pyplot as plt
import math

# 1

x = np.linspace(0, 10, 100)
y = np.arange(0, 10, 0.01)

# 2
x[0:3] # 

# 3
print('The first three entries of x are', x[0:3])

# 4

w = 10**(-np.linspace(1, 10, 10))
# the entries of w are 10^(-1), 10^(-2), ... , 10^(-10)

v = np.arange(1, len(w) + 1)

plt.semilogy(v, w)
plt.xlabel('v')
plt.ylabel('w')
plt.title('Question 4 Plot')
plt.show()

# 5

s = 3*w
plt.semilogy(v, w, label = 'v vs. w')
plt.semilogy(v, s, label = 'v vs. s')
plt.xlabel('v')
plt.ylabel('w and s')
plt.title('Question 5 Plot')
plt.legend()
plt.show()




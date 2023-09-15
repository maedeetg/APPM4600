import numpy as np
import math
import matplotlib.pyplot as plt
import random

# Problem 2c)

def driver1():
    f1 = lambda x: math.exp(x)
    x = 9.999999995000000*10**(-10)

    y1 = f1(x)

    return (y1 - 1)

# Problem 4a)

def driver2():
    f = lambda x: np.cos(x)
    t = np.linspace(0, np.pi, 31)
    y = f(t)

    N = len(t)

    tot = 0

    for k in range(0, N):
        tot += t[k]*y[k]

    print("The sum is: ", tot)
    return

# Problem 4b)

def driver3():
    theta = np.linspace(0, 2*np.pi, 100)

    x = lambda t: 1.2*(1 + 0.1*np.sin(15*t))*np.cos(t)

    y = lambda t: 1.2*(1 + 0.1*np.sin(15*t))*np.sin(t)

    xvals = x(theta)
    yvals = y(theta)

    plt.plot(xvals, yvals)
    plt.title('One Wavy Circle')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()
    
    return

def driver4():
    theta = np.linspace(0, 2*np.pi, 100)
    colors = range(10)

    for i in range(10):
        p = random.uniform(0, 2)

        x1 = lambda t: i*(1 + 0.05*np.sin((2 + i)*t + p))*np.cos(t)

        y1 = lambda t: i*(1 + 0.05*np.sin((2 + i)*t + p))*np.sin(t)

        plt.plot(x1(theta), y1(theta), colors[i])
        plt.title('10 Wavy Circles')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')

    plt.show()
    
    return

driver1()
driver2()
driver3()
driver4()
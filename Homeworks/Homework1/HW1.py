import numpy as np
import matplotlib.pyplot as plt
import math

# Problem 1
'''driver 1 will plot the necessary
   plots for problem 1 HW1'''

def driver1():
    poly1 = lambda x: x**9 - 18*x**8 + 144*x**7 - 672*x**6 + 2016*x**5 - 4032*x**4 + 5376*x**3 - 4608*x**2 + 2304*x - 512
    poly2 = lambda x: (x - 2)**9
    
    xvals = np.arange(1.920, 2.080, 0.001)
    yvals1 = poly1(xvals)
    yvals2 = poly2(xvals)
    
    plt.plot(xvals, yvals1)
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title('Plotting p(x)')
    plt.show()
    
    plt.plot(xvals, yvals2)
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title('Plotting p(x)')
    plt.show()
              
driver1()

# Problem 5
'''driver 2 will plot the necessary
   plots for problem 5b HW1'''    

def driver2():
    x_small = np.pi
    x_large = 10**6
    delta = np.array([10**(-16), 10**(-15), 10**(-14), 10**(-13), 10**(-12), 
                      10**(-11), 10**(-10), 10**(-9), 10**(-8), 10**(-7),
                      10**(-6), 10**(-5), 10**(-4), 10**(-3), 10**(-2), 
                      10**(-1), 10])

    f1 = lambda d: np.cos(x_small + d) - np.cos(x_small)
    g1 = lambda d: -2*np.sin((d/2) + x_small)*np.sin(d/2)
    f2 = lambda d: np.cos(x_large + d) - np.cos(x_large)
    g2 = lambda d: -2*np.sin((d/2) + x_large)*np.sin(d/2)

    f1_yvals_small = f1(delta)
    g1_yvals_small = g1(delta)
    f2_yvals_large = f2(delta)
    g2_yvals_large = g2(delta)

    plt.plot(delta, np.abs(g1_yvals_small - f1_yvals_small))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Delta')
    plt.ylabel('Absolute Error')
    plt.title('Plotting the Difference of Two Expressions for x = pi')
    plt.show()

    plt.plot(delta, np.abs(g2_yvals_large - f2_yvals_large))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Delta')
    plt.ylabel('Absolute Error')
    plt.title('Plotting the Difference of Two Expressions for x = 10**6')
    plt.show()
      
driver2()


# Problem 5 c
'''driver 3 will plot the necessary
   plots for problem 5b HW1'''   

def driver3():
    x_small = np.pi
    x_large = 10**6
    delta = np.array([10**(-16), 10**(-15), 10**(-14), 10**(-13), 10**(-12), 
                      10**(-11), 10**(-10), 10**(-9), 10**(-8), 10**(-7),
                      10**(-6), 10**(-5), 10**(-4), 10**(-3), 10**(-2), 
                      10**(-1), 10])

    f1 = lambda d: -2*np.sin((d/2) + x_small)*np.sin(d/2)
    g1 = lambda d: -d*np.sin(x_small) - ((d**2)/2)*np.cos(x_small)
    f2 = lambda d: -2*np.sin((d/2) + x_large)*np.sin(d/2)
    g2 = lambda d: -d*np.sin(x_large) - ((d**2)/2)*np.cos(x_large)

    f1_yvals_small = f1(delta)
    g1_yvals_small = g1(delta)
    f2_yvals_large = f2(delta)
    g2_yvals_large = g2(delta)

    plt.plot(delta, np.abs(g1_yvals_small - f1_yvals_small))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Delta')
    plt.ylabel('Absolute Error')
    plt.title('Plotting the Difference of Two Expressions for x = pi')
    plt.show()

    plt.plot(delta, np.abs(g2_yvals_large - f2_yvals_large))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Delta')
    plt.ylabel('Absolute Error')
    plt.title('Plotting the Difference of Two Expressions for x = 10**6')
    plt.show()
    
driver3()


    

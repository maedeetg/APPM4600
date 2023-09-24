# import libraries
import numpy as np
import math
    
def driver():
    f = lambda x: -np.sin(2*x) + (5 / 4)*x - (3 / 4)
    
    # There are 5 roots to approximate, so we will need 5 good guesses for
    # a starting point for each fixed point. To have a good initial guess, 
    # I plotted the fixed point iteration x_(n + 1) and y = x to determine 
    # a good point to start
    
    # intial points
    
    x1 = -0.899 # for first fixed point
    x2 = -0.898 # for second fixed point
    x3 = -0.544 # for third fixed point
    x4 = 2 # for fourth fixed point
    x5 = 4 # for fifth fixed point
    
    tol = 10**(-10)
    
    
    # for first fixed point
    
    [count, xstar, ier] = fixedpt(f, x1, tol, 100)
    print('The approximate fixed point for f:', xstar)
    print('f(xstar):', f(xstar))
    print('Error message reads:', ier)
    print('The number of iterations is:', count)
    
    # for second fixed point
    
    [count, xstar, ier] = fixedpt(f, x2, tol, 100)
    print('The approximate fixed point for f:', xstar)
    print('f(xstar):', f(xstar))
    print('Error message reads:', ier)
    print('The number of iterations is:', count)
    
    
    # for third fixed point
    
    [count, xstar, ier] = fixedpt(f, x3, tol, 100)
    print('The approximate fixed point for f:', xstar)
    print('f(xstar):', f(xstar))
    print('Error message reads:', ier)
    print('The number of iterations is:', count)
    
    # for fourth fixed point
    
    [count, xstar, ier] = fixedpt(f, x4, tol, 100)
    print('The approximate fixed point for f:', xstar)
    print('f(xstar):', f(xstar))
    print('Error message reads:', ier)
    print('The number of iterations is:', count)
    
    # for fifth fixed point
    
    [count, xstar, ier] = fixedpt(f, x5, tol, 100)
    print('The approximate fixed point for f:', xstar)
    print('f(xstar):', f(xstar))
    print('Error message reads:', ier)
    print('The number of iterations is:', count)
    
    
    return



# define routines
def fixedpt(f, x0, tol, Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    
    while (count < Nmax):
        
        count = count +1
        x1 = f(x0)
        
        if (abs(x1 - x0) < tol):
            xstar = x1
            ier = 0
            return [count, xstar, ier]
        x0 = x1

    xstar = x1
    ier = 1
    return [count, xstar, ier]
    

driver()
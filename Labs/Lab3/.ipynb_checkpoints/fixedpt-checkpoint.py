# import libraries
import numpy as np
import math
    
def driver1():

# functions
    fa = lambda x: x*(1 + ((7 - x**5) / (x**2)))**3
    fb = lambda x: x - ((x**5 - 7)/(x**2))
    fc = lambda x: x - ((x**5 - 7)/(5*x**4))
    fd = lambda x: x - ((x**5 - 7)/12)
    
    x1 = 7**(1/5)
  
    # VERIFY FIXED POINT
    # This should return (TRUE, TRUE, TRUE, TRUE) if x1 is a 
    # fixed point for the functions fa, fb, fc, and fd
   
    print (math.isclose(a = x1, b = fa(x1), rel_tol = 1e-6),
           math.isclose(a = x1, b = fb(x1), rel_tol = 1e-6),
           math.isclose(a = x1, b = fc(x1), rel_tol = 1e-6),
           math.isclose(a = x1, b = fd(x1), rel_tol = 1e-6))
    
    return

def driver2():
    fa = lambda x: x*(1 + ((7 - x**5) / (x**2)))**3
    fb = lambda x: x - ((x**5 - 7)/(x**2))
    fc = lambda x: x - ((x**5 - 7)/(5*x**4))
    fd = lambda x: x - ((x**5 - 7)/12)
    
    x0 = 1.0
    tol = 10**(-10)
    
    # test fa
    
    [xstar_a, ier] = fixedpt(fa, x0, tol, 2)
    print('The approximate fixed point for fa:', xstar_a)
    print('fa(xstar_a):', fa(xstar_a))
    print('Error message reads:', ier)
    
    # test fb
    
    [xstar_b, ier] = fixedpt(fb, x0, tol, 4)
    print('The approximate fixed point for fb:', xstar_b)
    print('fb(xstar_b):', fb(xstar_b))
    print('Error message reads:', ier)
    
    # test fc
    
    [xstar_c, ier] = fixedpt(fc, x0, tol, 500)
    print('The approximate fixed point for fc:', xstar_c)
    print('fc(xstar_c):', fc(xstar_c))
    print('Error message reads:', ier)
    
    # test fd
    
    [xstar_d, ier] = fixedpt(fd, x0, tol, 1000)
    print('The approximate fixed point for fd:', xstar_d)
    print('fd(xstar_d):', fd(xstar_d))
    print('Error message reads:', ier)
    
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
            return [xstar, ier]
        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier]
    

driver1()
driver2()
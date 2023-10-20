import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 

# Exercises 3.2

def driver1():
    f = lambda x: 1 / (1 + (10*x)**2)
    a = -1
    b = 1
    Neval = 1000
    xeval = np.linspace(a, b, Neval)
    Nint = 10
    
    yeval = eval_lin_spline(xeval, Neval, a, b, f, Nint)
    fex = np.zeros(Neval)
        
    for j in range(Neval):
        fex[j] = f(xeval[j]) 

    plt.figure()
    plt.plot(xeval,fex,'ro-')
    plt.plot(xeval,yeval,'bs-')
    plt.title("N = 10 Plot")
    plt.legend()
    plt.show()
     
    err = abs(yeval-fex)
    plt.figure()
    plt.plot(xeval,err,'ro-')
    plt.title("N = 10 Error Plot")
    plt.show()
    
def driver2():
    f = lambda x: 1 / (1 + (10*x)**2)
    a = -1
    b = 1
    Neval = 1000
    xeval = np.linspace(a, b, Neval)
    Nint = 15
    
    yeval = eval_lin_spline(xeval, Neval, a, b, f, Nint)
    fex = np.zeros(Neval)
        
    for j in range(Neval):
        fex[j] = f(xeval[j]) 

    plt.figure()
    plt.plot(xeval,fex,'ro-')
    plt.plot(xeval,yeval,'bs-')
    plt.title("N = 15 Plot")
    plt.legend()
    plt.show()
     
    err = abs(yeval-fex)
    plt.figure()
    plt.plot(xeval,err,'ro-')
    plt.title("N = 15 Error Plot")
    plt.show()
    
def driver3():
    f = lambda x: 1 / (1 + (10*x)**2)
    a = -1
    b = 1
    Neval = 1000
    xeval = np.linspace(a, b, Neval)
    Nint = 20
    
    yeval = eval_lin_spline(xeval, Neval, a, b, f, Nint)
    fex = np.zeros(Neval)
        
    for j in range(Neval):
        fex[j] = f(xeval[j]) 

    plt.figure()
    plt.plot(xeval,fex,'ro-')
    plt.plot(xeval,yeval,'bs-')
    plt.title("N = 20 Plot")
    plt.legend()
    plt.show()
     
    err = abs(yeval-fex)
    plt.figure()
    plt.plot(xeval,err,'ro-')
    plt.title("N = 20 Error Plot")
    plt.show()
    
def eval_lin_spline(xeval, Neval, a, b, f, Nint):

    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a, b, Nint + 1)
   
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval) 
    
    for jint in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        '''let n denote the length of ind'''
        
        '''temporarily store your info for creating a line in the interval of 
         interest'''
        
        ind = find_point(xeval, xint[jint], xint[jint + 1])
        n = len(ind)
        
        a1 = xint[jint]
        fa1 = f(a1)
        b1 = xint[jint + 1]
        fb1 = f(b1)
        
        
        for kk in range(n):
            '''use your line evaluator to evaluate the lines at each of the points in the interval'''
            '''yeval(ind(kk)) = call your line evaluator at xeval(ind(kk)) with 
            the points (a1,fa1) and (b1,fb1)'''
            yeval[ind[kk]] = line(a1, b1, fa1, fb1, xeval[ind[kk]])
            
            
    return yeval
        
def find_point(xeval, a, b):
    return np.where((xeval <= b) & (xeval >= a))[0]

# Problem 2

def line(x0, x1, f0, f1, x):
    numer = f1 - f0
    denom = x1 - x0

    slope = numer / denom
    
    f = slope*(x - x1) + f1
    
    return f

# if __name__ == '__main__':
#       # run the drivers only if this is called from the command line
#         driver()

driver1()
driver2()
driver3()
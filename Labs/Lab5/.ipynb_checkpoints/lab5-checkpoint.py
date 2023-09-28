import numpy as np
import math

def driver():
    
    a = 2
    b = 4.5
    x0 = 4.5
    tol = 10**(-10)
    Nmax = 100
    f = lambda x: np.exp(x**2 + 7*x - 30) - 1
    df = lambda x: np.exp(x**2 + 7*x - 30) * (2*x + 7)
    
    print("Root Finding Using Bisection")
    print("")
    [astar, ier, count] = bisection(f, a, b, tol)
    print('The approximate root is', astar)
    print('the error message reads:', ier)
    print('f(astar) =', f(astar))
    print('The number of iterations was', count)
    print("")
    
    print("Root Finding Using Newton's Method")
    print("")
    [p,pstar,info,it] = newton(f, df, x0, tol, Nmax)
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)
    print("")
    
    print("Root Finding Using Hybrid Method")
    print("")
    bis_new(f, df, a, b, tol)
    print("")

def bisection(f, a, b, tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b);
    
    if (fa*fb > 0):
        ier = 1
        astar = a
        return [astar, ier]

#   verify end points are not a root 
    if (fa == 0):
        astar = a
        ier =0
        return [astar, ier]

    if (fb ==0):
        astar = b
        ier = 0
        return [astar, ier]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
        fd = f(d)
        if (fd ==0):
            astar = d
            ier = 0
            return [astar, ier]
        
        if (fa*fd<0):
            b = d
        else: 
            a = d
            fa = fd
            
        d = 0.5*(a+b)
        count = count +1
        
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier, count]

# define routines
def bis_new(f, df, a, b, tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b);
    
    if (fa*fb > 0):
        ier = 1
        astar = a
        return [astar, ier]

#   verify end points are not a root 
    if (fa == 0):
        astar = a
        ier =0
        return [astar, ier]

    if (fb == 0):
        astar = b
        ier = 0
        return [astar, ier]

    count = 0
    c = 0.5*(a+b)
    
    # if the fixed point iteration is in the interval, then
    # continue with newton's method
    if (c - (f(c) / df(c)) < b) & (c - (f(c) / df(c)) > b):
        
        [p, pstar, info, it] = newton(f, df, c, tol, 100)
        print('the approximate root is', '%16.16e' % pstar)
        print('the error message reads:', '%d' % info)
        print('Number of iterations:', '%d' % it)
    
    while (abs(c-a) > tol):
        fc = f(c)
        if (fc ==0):
            astar = c
            ier = 0
            return [astar, ier]
        
        if (fa*fc<0):
            b = c
        else: 
            a = c
            fa = fc
            
        c = 0.5*(a+b)
        count = count +1
        
#      print('abs(d-a) = ', abs(d-a))
      
    astar = c
    ier = 0
    
    print('The approximate root is', astar)
    print('the error message reads:', ier)
    print('f(astar) =', f(astar))
    print('The number of iterations is', count)
    
    return 

def newton(f, fp, p0, tol, Nmax):
    """"" Newton iteration.
  
  Inputs:
    f,fp - function and derivative
    p0   - initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    pstar - the last iterate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail) """
    
    p = np.zeros(Nmax+1)
    
    p[0] = p0
    
    for it in range(Nmax):
        
        p1 = p0-f(p0)/fp(p0)
        p[it+1] = p1
        
        if (abs(p1-p0) < tol):
            
            pstar = p1
            info = 0
            return [p,pstar,info,it]
        
        p0 = p1
        
    pstar = p1
    info = 1
    
    return [p,pstar,info,it]
      
driver()

 
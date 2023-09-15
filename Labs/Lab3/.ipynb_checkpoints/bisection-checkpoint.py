# import libraries
import numpy as np

# Problem 1
def driver1():

# use routines    
    f_1a = lambda x: x**2*(x - 1)
    a1 = 0.5
    b1 = 2
    a2 = -1
    b2 = 0.5
    a3 = -1
    b3 = 2
    tol = 1e-7
    
    [astar,ier] = bisection(f_1a, a1, b1, tol)
    print('The approximate root for problem a) is',astar)
    print('the error message reads:',ier)
    print('f_1a(astar) =', f_1a(astar))
    
    [astar,ier] = bisection(f_1a, a2, b2, tol)
    print('The approximate root for problem b) is',astar)
    print('the error message reads:',ier)
    print('f_1a(astar) =', f_1a(astar))
    
    [astar,ier] = bisection(f_1a, a3, b3, tol)
    print('The approximate root for problem c) is',astar)
    print('the error message reads:',ier)
    print('f_1a(astar) =', f_1a(astar))
    
    return

def driver2():
    tol = 10**(-5)
    f_2a = lambda x: (x - 1)*(x - 3)*(x - 5)
    a_2a = 0
    b_2a = 2.4

    f_2b = lambda x: (x - 1)**2*(x - 3)
    a_2b = 0
    b_2b = 2
    
    f_2c = lambda x: np.sin(x)
    a_2c = 0
    b_2c = 0.1
    a_2c_ = 0.5
    b_2c_ = (3*np.pi) / 4
    
    [astar_a, ier] = bisection(f_2a, a_2a, b_2a, tol)
    print('The approximate root for problem a) is', astar_a)
    print('the error message reads:',ier)
    print('f_2a(astar) =', f_2a(astar_a))
    
    [astar_b, ier] = bisection(f_2b, a_2b, b_2b, tol)
    print('The approximate root for problem b) is',astar_b)
    print('the error message reads:',ier)
    print('f_2b(astar) =', f_2b(astar_b))
    
    [astar_c, ier] = bisection(f_2c, a_2c, b_2c, tol)
    print('The approximate root for problem c) is',astar_c)
    print('the error message reads:',ier)
    print('f_2c(astar) =', f_2c(astar_c))
    
    [astar_d, ier] = bisection(f_2c, a_2c_, b_2c_, tol)
    print('The approximate root for problem c) with new interval is',astar_d)
    print('the error message reads:',ier)
    print('f_2c(astar) =', f_2c(astar_d))
    
    return


# define routines
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
    return [astar, ier]
      
driver1()        
driver2()


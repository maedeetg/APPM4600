# import libraries
import numpy as np
import matplotlib.pyplot as plt

# Problem 1c
def driver1():

# use routines    
    f_1 = lambda x: np.sin(x) - 2*x + 1
    a = 0
    b = np.pi / 2
    tol = 10**(-8)
    
    [astar, ier, count] = bisection_count(f_1, a, b, tol)
    print('The approximate root for problem a) is', astar)
    print('the error message reads:', ier)
    print('f_1(astar) =', f_1(astar))
    print('The number of iterations used was', count)
    
    return

# Problem 2a
def driver2():

# use routines    
    f_1 = lambda x: (x - 5)**9
    f_2 = lambda x: x**9 - 45*x**8 + 900*x**7 - 10500*x**6 + 78750*x**5 - 393750*x**4 + 1312500*x**3 - 2812500*x**2 + 3515625*x - 1953125
    a = 4.82
    b = 5.2
    tol = 10**(-4)
    
    [astar, ier] = bisection(f_1, a, b, tol)
    print('The approximate root for problem a) is', astar)
    print('the error message reads:', ier)
    print('f_1(astar) =', f_1(astar))
    
    [astar, ier] = bisection(f_2, a, b, tol)
    print('The approximate root for problem a) is', astar)
    print('the error message reads:', ier)
    print('f_1(astar) =', f_1(astar))
    
    return

# Problem 3b
def driver3():

# use routines    
    f_1 = lambda x: x**3 + x - 7
    a = 1
    b = 4
    tol = 10**(-3)
    
    [astar, ier, count] = bisection_count(f_1, a, b, tol)
    print('The approximate root for problem a) is', astar)
    print('the error message reads:', ier)
    print('f_1(astar) =', f_1(astar))
    print('The number of iterations used was', count)
    
    return

# Problem 5a
def driver4():
    f = lambda x: x - 4*np.sin(2*x) - 3
    xvals = np.linspace(-2*np.pi, 2*np.pi, 500)
    yvals = f(xvals)
    
    plt.plot(xvals, yvals)
    plt.axhline(y = 0, color = 'black', linestyle = 'dashed')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Problem 5a Graph')
    plt.show()
    
    return
    

# # define routines

def bisection_count(f, a, b, tol):
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
driver3()
driver4()


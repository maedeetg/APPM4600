import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from scipy import special

# Problem 1

def driver1():
    
    # Problem 1a
    Ti = 20
    Ts = -15
    a = 0.138*10**(-6)
    eps = 10**(-13)
    t = 60*60*24*60
    
    T = lambda x: (Ti - Ts)*special.erf(x / (2 * (a*t)**(1/2))) + Ts
    
    xvals = np.linspace(0, 1)
    yvals = T(xvals)
    
    plt.plot(xvals, yvals)
    plt.title('Question 1a Plot')
    plt.xlabel('x')
    plt.ylabel('T(x, t)')
    plt.show()
    
    # Problem 1b
    
    a0 = 0
    b0 = 1
    tol = 10**(-13)
    
    [astar, ier] = bisection(T, a0, b0, tol)
    print('The approximate root is', astar)
    print('the error message reads:', ier)
    print('T(astar) =', T(astar))
    
    # Problem 1c
    
    x0 = 0.01
    x1 = 1
    tol = 10**(-13)
    Nmax = 100
    
    #dT = lambda x: (Ti - Ts)*(2 / (np.pi)**(1/2))*(1 / (2*()))
    dT = lambda x: (Ti - Ts)*(2 / (np.pi)**(1/2))*(1 / 2*((a*t)**(1/2)))*np.exp(-(x / (2 * (a*t)**(1/2)))**(2))
    
    [p, pstar, info, it] = newton(T, dT, x0, tol, Nmax)
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)
    
    [p, pstar, info, it] = newton(T, dT, x1, tol, Nmax)
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)
    
    return

# Problem 4i

def driver2():
    f = lambda x: (np.exp(x) -3*x**2)**3
    df = lambda x: 3*(np.exp(x) - 3*x**2)**2*(np.exp(x) - 6*x)
    x0 = 4
    tol = 10**(-10)
    Nmax = 100
    
    [p, pstar, info, it] = newton(f, df, x0, tol, Nmax)
    print('the list of each value at each iteration', p)
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)

# Problem 4ii

def driver3():
    g = lambda x: (np.exp(x) - 3*x**2) / (3*np.exp(x) - 18*x)
    dg = lambda x: (6*x**2 + np.exp(x)*(2 + (x - 4)*x)) / (np.exp(x) - 6*x)**2
    x0 = 4
    tol = 10**(-10)
    Nmax = 100
    
    [p, pstar, info, it] = newton(g, dg, x0, tol, Nmax)
    print('the list of each value at each iteration', p)
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)
    
# Problem 4iii

def driver4():
    f = lambda x: (np.exp(x) -3*x**2)**3
    df = lambda x: 3*(np.exp(x) - 3*x**2)**2*(np.exp(x) - 6*x)
    x0 = 4
    tol = 10**(-10)
    Nmax = 100
    
    [p, pstar, info, it] = adj_newton(f, df, x0, tol, Nmax)
    print('the list of each value at each iteration', p)
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)
    
# Problem 5 secant

def driver5():
    f = lambda x: x**6 - x - 1
    x0 = 2
    x1 = 1
    tol = 10**(-10)
    Nmax = 100
    
    [error_lst, x, x1, ier] = secant(f, x0, x1, tol, Nmax)
    print("The error of each value at each iteration is", error_lst)
    print("The approximate root is", x1)
    print("The error message reads", ier)
    
# Problem 5 Newton
    
def driver6():
    f = lambda x: x**6 - x - 1
    df = lambda x: 6*x**5 - 1
    x0 = 2
    tol = 10**(-10)
    Nmax = 100
    
    [error_lst, p, pstar, info, it] = adj_newton2(f, df, x0, tol, Nmax)
    print('the error of each value at each iteration', error_lst)
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)
    
# Problem 5 b

def driver7():
    f = lambda x: x**6 - x - 1
    x0 = 2
    x1 = 1
    tol = 10**(-10)
    Nmax = 100
    
    f = lambda x: x**6 - x - 1
    df = lambda x: 6*x**5 - 1
    x0 = 2
    tol = 10**(-10)
    Nmax = 100
    
    [error_lst1, p, pstar, info, it] = adj_newton2(f, df, x0, tol, Nmax)
    
    [error_lst2, x, x1, ier] = secant(f, x0, x1, tol, Nmax)
    
    error1 = error_lst1[1:]
    error1.append(0)
    error2 = error_lst2[1:]
    error2.append(0)
    # print ('k1', error1)
    # print('k2', error_lst1)
    
    plt.loglog(error_lst1, error1)
    plt.title("Newtons")
    plt.show()
    
    plt.loglog(error_lst2, error2)
    plt.title("Secant")
    plt.show()
    
    return
    
    
    
    
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

def adj_newton(f, fp, p0, tol, Nmax):
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
        
        p1 = p0-3*f(p0)/fp(p0)
        p[it+1] = p1
        
        if (abs(p1-p0) < tol):
            
            pstar = p1
            info = 0
            return [p,pstar,info,it]
        
        p0 = p1
        
    pstar = p1
    info = 1
    
    return [p,pstar,info,it]

def adj_newton2(f, fp, p0, tol, Nmax):
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
    
    p = np.zeros(Nmax + 1)
    
    p[0] = p0
    
    for it in range(Nmax):
        
        p1 = p0-f(p0)/fp(p0)
        p[it+1] = p1
        
        if (abs(p1-p0) < tol):
            
            pstar = p1
            info = 0
            error_lst = [np.abs(i - pstar) for i in p]
            return [error_lst, p,pstar,info,it]
        
        p0 = p1
        
    pstar = p1
    info = 1
    error_lst = [np.abs(i - pstar) for i in p]
    return [error_lst, p, pstar, info, it]

def secant(f, x0, x1, tol, Nmax):
    
    if (f(x0) == 0):
        return [x0, 0]
    
    if (f(x1) == 0):
        return [x1, 0]
    
    x = []
    x.append(x0)
    x.append(x1)
    
    fx1 = f(x1)
    fx0 = f(x0)
    
    for i in range(0, Nmax):
        if (abs(fx1 - fx0) == 0):
            ier = 1
            return [x, x1, ier]
        
        x2 = x1 - f(x1)*(x1-x0)/(f(x1) - f(x0))
        x.append(x2)
        
        if (abs(x2 - x1) < tol):
            ier = 0
            error_lst = [np.abs(i - x2) for i in x]
            return [error_lst, x, x2, ier]
        
        x0 = x1
        fx0 = fx1
        x1 = x2
        fx1 = f(x2)
        
    ier = 0
    
    error_lst = [np.abs(i - x2) for i in x]
    return [error_lst, x, x2, 0]


#driver1()
#driver2()
#driver3()
#driver4()
#driver5()
#driver6()
driver7()
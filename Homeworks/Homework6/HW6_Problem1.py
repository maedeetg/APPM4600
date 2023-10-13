import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 

# Problem 1, Lazy newton

def driver1():
    print("LAZY NEWTON")
    print(' ')
    # Initial guess i)
    x1 = np.array([1, 1])
    tol = 10**(-10)
    Nmax = 50
    
    [xstar1, ier1, its1] = LazyNewton(x1, tol, Nmax)
    print("Starting guess i)")
    print('')
    print("The solution is:", xstar1)
    print("The error message reads:", ier1)
    print("The number of iterations was:", its1)
    print(" ")
    
    # Initial guess ii)
    x1 = np.array([1, -1])
    tol = 10**(-10)
    Nmax = 50
    
    [xstar2, ier2, its2] = LazyNewton(x1, tol, Nmax)
    
    print("Starting guess ii)")
    print('')
    print("The solution is:", xstar2)
    print("The error message reads:", ier2)
    print("The number of iterations was:", its2)
    print(" ")
    
# Probelm 1, Broyden
def driver2():
    print("BROYDEN")
    print('')
    # Initial guess i)
    x1 = np.array([1, 1])
    tol = 10**(-10)
    Nmax = 50
    
    [xstar1, ier1, its1] = Broyden(x1, tol, Nmax)
    print("Starting guess i)")
    print('')
    print("The solution is:", xstar1)
    print("The error message reads:", ier1)
    print("The number of iterations was:", its1)
    print(" ")
    
    # Initial guess ii)
    x2 = np.array([1, -1])
    tol = 10**(-10)
    Nmax = 50
    
    [xstar2, ier2, its2] = Broyden(x2, tol, Nmax)
    
    print("Starting guess ii)")
    print('')
    print("The solution is:", xstar2)
    print("The error message reads:", ier2)
    print("The number of iterations was:", its2)
    print(" ")
     
# Problem 1, Newton
def driver3():
    print("NEWTON")
    print(' ')
     # Initial guess i)
    x1 = np.array([1, 1])
    tol = 10**(-10)
    Nmax = 50
    
    [xstar1, ier1, its1] = Newton(x1, tol, Nmax)
    print("Starting guess i)")
    print('')
    print("The solution is:", xstar1)
    print("The error message reads:", ier1)
    print("The number of iterations was:", its1)
    print(" ")
    
    # Initial guess ii)
    x2 = np.array([1, -1])
    tol = 10**(-10)
    Nmax = 50
    
    [xstar2, ier2, its2] = Newton(x2, tol, Nmax)
    
    print("Starting guess ii)")
    print('')
    print("The solution is:", xstar2)
    print("The error message reads:", ier2)
    print("The number of iterations was:", its2)
    print(" ")
    
def evalF(x0): 
    x = x0[0]
    y = x0[1]

    F = np.zeros(2)
    
    F[0] = x**2 + y**2 - 4
    F[1] = np.exp(x) + y - 1
    
    return F
    
def evalJ(x0): 
    f = lambda x, y: x**2 + y**2 - 4
    g = lambda x, y: np.exp(x) + y - 1
    
    df_x = lambda x, y: 2*x
    df_y = lambda x, y: 2*y
    dg_x = lambda x, y: np.exp(x)
    dg_y = lambda x, y: 1
    
    x = x0[0]
    y = x0[1]
    
    J = np.array([[df_x(x, y), df_y(x, y)], [dg_x(x, y), dg_y(x, y)]])
    
    return J

def Newton(x0,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
        J = evalJ(x0)
        Jinv = inv(J)
        F = evalF(x0)
       
        x1 = x0 - Jinv.dot(F)
       
        if (norm(x1-x0) < tol):
            xstar = x1
            ier =0
            return[xstar, ier, its]
           
        x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]

def LazyNewton(x0,tol,Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    J = evalJ(x0)
    Jinv = inv(J)
    for its in range(Nmax):

        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)
       
        if (norm(x1-x0) < tol):
            xstar = x1
            ier =0
            return[xstar, ier,its]
           
        x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]   
    
def Broyden(x0,tol,Nmax):
    '''tol = desired accuracy
    Nmax = max number of iterations'''

    '''Sherman-Morrison 
   (A+xy^T)^{-1} = A^{-1}-1/p*(A^{-1}xy^TA^{-1})
    where p = 1+y^TA^{-1}Ax'''

    '''In Newton
    x_k+1 = xk -(G(x_k))^{-1}*F(x_k)'''


    '''In Broyden 
    x = [F(xk)-F(xk-1)-\hat{G}_k-1(xk-xk-1)
    y = x_k-x_k-1/||x_k-x_k-1||^2'''

    ''' implemented as in equation (10.16) on page 650 of text'''
    
    '''initialize with 1 newton step'''
    
    A0 = evalJ(x0)

    v = evalF(x0)
    A = np.linalg.inv(A0)

    s = -A.dot(v)
    xk = x0+s
    for  its in range(Nmax):
        '''(save v from previous step)'''
        w = v
        ''' create new v'''
        v = evalF(xk)
        '''y_k = F(xk)-F(xk-1)'''
        y = v-w;                   
        '''-A_{k-1}^{-1}y_k'''
        z = -A.dot(y)
        ''' p = s_k^tA_{k-1}^{-1}y_k'''
        p = -np.dot(s,z)                 
        u = np.dot(s,A) 
        ''' A = A_k^{-1} via Morrison formula'''
        tmp = s+z
        tmp2 = np.outer(tmp,u)
        A = A+1./p*tmp2
        ''' -A_k^{-1}F(x_k)'''
        s = -A.dot(v)
        xk = xk+s
        if (norm(s)<tol):
            alpha = xk
            ier = 0
            return[alpha,ier,its]
    alpha = xk
    ier = 1
    return[alpha,ier,its]  
        
driver1()
driver2()
driver3()

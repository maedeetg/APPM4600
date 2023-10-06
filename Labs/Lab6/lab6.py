import numpy as np
import numpy.linalg
from numpy.linalg import inv 
from numpy.linalg import norm 
from numpy.linalg import det

# Prelab

def driver1():
    f = lambda x: np.cos(x)
    x = np.pi / 2
    act = -np.sin(x)
    
    h = 0.01*2.**(-np.arange(0, 10))
    deriv = (f(x + h) - f(x)) / h
    
    
    errs = abs(deriv - act)
    
    print("The forward difference approximations:", deriv)
    print("The absoulte error:", errs)
    print('')
    return
    
def driver2():
    f = lambda x: np.cos(x)
    x = np.pi / 2
    act = -np.sin(x)
    
    h = 0.01*2.**(-np.arange(0, 10))
    deriv = (f(x + h) - f(x - h)) / (2*h)
    
    errs = abs(deriv - act)
    
    print("The backward difference approximations:", deriv)
    print("The absoulte error:", errs)
    print('')
    return

# Slacker newton

def driver3():
    x0 = np.array([[1], [0]])
    tol1 = 10**(-10)
    tol2 = 10**(0)
    Nmax = 50
    [xstar, ier, its] = SlackerNewton(x0, tol1, tol2, Nmax)
    
    print("The estimated root is:", xstar)
    print("The error message reads:", ier)
    print("The number of iterations is:", its)
    return

# Jacobian approx

def driver4():
    x0 = np.array([[1], [0]])
    tol = 10**(-10)
    Nmax = 50
    h_val1 = 10**(-7)
    h_val2 = 10**(-3)
    [ier1, it1, pstar1] = approx_jac(x0, tol, Nmax, h_val1)
    [ier2, it2, pstar2] = approx_jac(x0, tol, Nmax, h_val2)
    
    print("For h_val = ", h_val1)
    print("The estimated root is:", pstar1)
    print("The error message reads:", ier1)
    print("The number of iterations is:", it1)
    
    print("For h_val = ", h_val2)
    print("The estimated root is:", pstar2)
    print("The error message reads:", ier2)
    print("The number of iterations is:", it2)
    return

# Hybrid

def driver5():
    x0 = np.array([[1], [0]])
    tol1 = 10**(-10)
    tol2 = 10**(0)
    Nmax = 50
    h_val = 10**(-3)
    [xstar, ier, its]  = hybrid(x0, tol1, tol2, Nmax, h_val)
    
    print("For h_val = ", h_val)
    print("The estimated root is:", xstar)
    print("The error message reads:", ier)
    print("The number of iterations is:", its)
    return

# Methods

def jacobian():
    f = lambda x1, x2: 4*x1**2 + x2**2 - 4 
    g = lambda x1, x2: x1 + x2 - np.sin(x1 - x2) 
    
    df_x1 = lambda x1, x2: 8*x1
    df_x2 = lambda x1, x2: 2*x2
    dg_x1 = lambda x1, x2: 1 - np.cos(x1 - x2)
    dg_x2 = lambda x1, x2: 1 + np.cos(x1 - x2)
    
    return [df_x1, df_x2, dg_x1, dg_x2]
    
def jacobian_eval(J, x0):
    x = x0[0][0]
    y = x0[1][0]
    
    [df_x1, df_x2, dg_x1, dg_x2] = J
    
    J = np.array([[df_x1(x, y), -df_x2(x, y)], [-dg_x1(x, y), dg_x2(x, y)]])
    J_inv = inv(J)
    
    return J_inv

def evalF(x0):
    f = lambda x1, x2: 4*x1**2 + x2**2 - 4
    g = lambda x1, x2: x1 + x2 - np.sin(x1 - x2)
    
    x1 = x0[0][0]
    x2 = x0[1][0]
    
    return np.array([[f(x1, x2)], [g(x1, x2)]])

# Slacker Newton

# My condition for recomputing the Jacobian will be if the distance between two iterates
# is growing. So I will store the distance between the two iterates and if the distance
# is larger than 10**(0), then I will recompute the Jacobian

def SlackerNewton(x0, tol1, tol2, Nmax):
    J = jacobian()
    Jinv = jacobian_eval(J, x0)
    
    for its in range(Nmax):

        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)
        
        # check if the distance of the iterates is larger than 10**(0)
        
        if (norm(x1 - x0) >= tol2):
            # recompute the Jacobian
            print("Jacobian recomputed")
            Jinv = jacobian_eval(J, x1)
            continue
       
        elif (norm(x1 - x0) < tol1):
            xstar = x1
            ier = 0
            return[xstar, ier, its]
           
        x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar, ier, its] 

# 3.3 Exercises

def jacobian_est(x0, h_val):
    # I will be using backward difference
    x1 = x0[0][0]
    x2 = x0[1][0]
    
    a = np.zeros((len(x0), 1))
    a[0][0] = x1 + h_val
    a[1][0] = x2
    
    b = np.zeros((len(x0), 1))
    b[0][0] = x1 - h_val
    b[1][0] = x2
    
    c = np.zeros((len(x0), 1))
    c[0][0] = x1
    c[1][0] = x2 + h_val
    
    d = np.zeros((len(x0), 1))
    d[0][0] = x1 
    d[1][0] = x2 - h_val
    
    J_x = (evalF(a) - evalF(x0)) / (h_val)
    J_y = (evalF(c) - evalF(x0)) / (h_val)
    
    J = np.zeros((2, 2))
    J[0][0] = J_x[0][0]
    J[0][1] = J_y[0][0]
    J[1][0] = J_x[1][0]
    J[1][1] = J_y[1][0]

    return J

def approx_jac(x0, tol, Nmax, h_val):
    vals = []
    
    for i in range(Nmax):
        # function vector
        f_vec = evalF(x0)
        
        # Jacobian matrix (2x2)
        h = h_val*norm(x0)
        Jac = inv(jacobian_est(x0, h))
        
        # x_n+1 and y_n+1
        new = x0 - Jac@f_vec
        
        if (norm(new - x0, 2) < tol):
            vals.append(new)
            pstar = new
            ier = 0
            it = i
            
            return [ier, it, pstar]
        
        x0 = new
    
    pstar = new
    vals.append(new)
    ier = 0
    it = Nmax
    
    return [ier, it, pstar]

# 3.4 Exercises

def hybrid(x0, tol1, tol2, Nmax, h_val):
    h = h_val*norm(x0)
    J = inv(jacobian_est(x0, h))
    
    for its in range(Nmax):

        F = evalF(x0)
        x1 = x0 - J.dot(F)
        
        # check if the distance of the iterates is larger than 10**(0)
        
        if (norm(x1 - x0) >= tol2):
            # recompute the Jacobian
            print("Jacobian recomputed")
            h = (1/2)*h
            J = inv(jacobian_est(x1, h))
            continue
       
        elif (norm(x1 - x0) < tol1):
            xstar = x1
            ier = 0
            return[xstar, ier, its]
           
        x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar, ier, its] 

driver1()
driver2()
driver3()
driver4()
driver5()

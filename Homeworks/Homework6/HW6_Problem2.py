import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 

# Problem 2, Newton

def driver1():
    print("NEWTON")
    print("")
    
    x0 = np.array([0, 0, 0])
    tol = 10**(-6)
    Nmax = 50
    
    [xstar, ier, its] = Newton(x0, tol, Nmax)
    
    print("The estimated solution is:", xstar)
    print("The error message reads:", ier)
    print("The number of iterations needed:", its)
    print(" ")
    
    return 

def driver2():
    print("STEEPEST DESCENT")
    print('')
    
    x0 = np.array([0, 0, 0])
    tol = 10**(-6)
    Nmax = 50
    
    [xstar, gval, ier, its] = SteepestDescent(x0, tol, Nmax)
    print("The steepest descent code found the solution ", xstar)
    print("g evaluated at this point is ", gval)
    print("The error message reads:", ier)
    print("The number of iterations needed:", its)
    print(" ")
    
def driver3():
    print("NEWTON + STEEPEST")
    print("")
    
    x1 = np.array([0, 0, 0])
    tol1 = 5*10**(-2)
    Nmax1 = 50
    
    [xstar1, g1, ier1, its] = SteepestDescent(x1, tol1, Nmax1)
    
    x2 = xstar1
    tol2 = 10**(-6)
    Nmax2 = 50
    
    [xstar2, ier2, its2] = Newton(x2, tol2, Nmax2)
    print("The estimated solution is:", xstar2)
    print("The error message reads:", ier2)
    print("The number of iterations needed for entire process:", its + its2)
    print(" ")
    
    

def evalF(x0):
    x = x0[0]
    y = x0[1]
    z = x0[2]

    F = np.zeros(3)
    
    F[0] = x + np.cos(x*y*z) - 1
    F[1] = (1 - x)**(1/4) + y + 0.05*z**2 - 0.15*z - 1
    F[2] = -x**2 - 0.1*y**2 + 0.01*y + z - 1
    
    return F

def evalJ(x0): 
    f = lambda x, y, z: x + np.cos(x*y*z) - 1
    g = lambda x, y, z: (1 - x)**(1/4) + y + 0.05*z**2 - 0.15*z - 1
    h = lambda x, y, z: -x**2 - 0.1*y**2 + 0.01*y + z - 1
    
    df_x = lambda x, y, z: 1 - np.sin(x*y*z)*y*z
    df_y = lambda x, y, z: -np.sin(x*y*z)*x*z
    df_z = lambda x, y, z: -np.sin(x*y*z)*x*y
    dg_x = lambda x, y, z: (1/4)*(1 - x)**(-3/4)*(-1)
    dg_y = lambda x, y, z: 1
    dg_z = lambda x, y, z: 0.1*z - 0.15
    dh_x = lambda x, y, z: -2*x
    dh_y = lambda x, y, z: -0.2*y + 0.01
    dh_z = lambda x, y, z: 1
    
    
    x = x0[0]
    y = x0[1]
    z = x0[2]
    
    J = np.array([[df_x(x, y, z), df_y(x, y, z), df_z(x, y, z)], 
                  [dg_x(x, y, z), dg_y(x, y, z), dg_z(x, y, z)], 
                  [dh_x(x, y, z), dh_y(x, y, z), dh_z(x, y, z)]])
    
    return J

def evalg(x0):

    F = evalF(x0)
    g = F[0]**2 + F[1]**2 + F[2]**2
    return g

def eval_gradg(x0):
    F = evalF(x0)
    J = evalJ(x0)
    
    gradg = np.transpose(J).dot(F)
    return gradg

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
    return [xstar,ier,its]

def SteepestDescent(x0, tol, Nmax):
    
    for its in range(Nmax):
        g1 = evalg(x0)
        z = eval_gradg(x0)
        z0 = norm(z)

        if z0 == 0:
            print("Zero gradient")
        z = z/z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x0 - alpha3*z
        g3 = evalg(dif_vec)

        while g3>=g1:
            alpha3 = alpha3/2
            dif_vec = x0 - alpha3*z
            g3 = evalg(dif_vec)
            
        if alpha3<tol:
            print("No likely improvement")
            ier = 0
            return [x0, g1, ier, its]
        
        alpha2 = alpha3/2
        dif_vec = x0 - alpha2*z
        g2 = evalg(dif_vec)

        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3

        alpha0 = 0.5*(alpha2 - h1/h3)
        dif_vec = x0 - alpha0*z
        g0 = evalg(dif_vec)

        if g0<=g3:
            alpha = alpha0
            gval = g0

        else:
            alpha = alpha3
            gval =g3

        x0 = x0 - alpha*z

        if abs(gval - g1)<tol:
            ier = 0
            return [x0, gval, ier, its]

    print('max iterations exceeded')    
    ier = 1        
    return [x0, g1, ier, its]


driver1()
driver2()
driver3()


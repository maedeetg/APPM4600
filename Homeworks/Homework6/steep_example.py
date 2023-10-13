#libraries:
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm


def driver():

    Nmax = 100
    x0= np.array([0,0,1])
    tol = 1e-6
    
    [xstar,gval,ier] = SteepestDescent(x0,tol,Nmax)
    print("the steepest descent code found the solution ",xstar)
    print("g evaluated at this point is ", gval)
    print("ier is ", ier	)

###########################################################
#functions:
def evalF(x0):
    x = x0[0]
    y = x0[1]
    z = x0[2]

    F = np.zeros(3)
    
    F[0] = x + np.cos(x*y*z) - 1
    F[1] = (1 - x)**(1/4) + y + 0.05*z**2 - 0.15*z - 1
    F[2] = -x**2 - 0.1*y**2 + 0.01*y + z - 1
    
    return F

def evalJ(x): 
    f = lambda x, y, z: x + np.cos(x*y*z) - 1
    g = lambda x, y, z: (1 - x)**(1/4) + y + 0.05*z**2 - 0.15*z - 1
    h = lambda x, y, z: -x**2 - 0.1*y**2 + 0.01*y + z - 1
    
    df_x = lambda x, y, z: -np.sin(x*y*z)*y*z
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
    z = x0[0]
    
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


###############################
### steepest descent code

def SteepestDescent(x,tol,Nmax):
    
    for its in range(Nmax):
        g1 = evalg(x)
        z = eval_gradg(x)
        z0 = norm(z)

        if z0 == 0:
            print("zero gradient")
        z = z/z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3*z
        g3 = evalg(dif_vec)

        while g3>=g1:
            alpha3 = alpha3/2
            dif_vec = x - alpha3*z
            g3 = evalg(dif_vec)
            
        if alpha3<tol:
            print("no likely improvement")
            ier = 0
            return [x,g1,ier]
        
        alpha2 = alpha3/2
        dif_vec = x - alpha2*z
        g2 = evalg(dif_vec)

        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3

        alpha0 = 0.5*(alpha2 - h1/h3)
        dif_vec = x - alpha0*z
        g0 = evalg(dif_vec)

        if g0<=g3:
            alpha = alpha0
            gval = g0

        else:
            alpha = alpha3
            gval =g3

        x = x - alpha*z

        if abs(gval - g1)<tol:
            ier = 0
            return [x,gval,ier]

    print('max iterations exceeded')    
    ier = 1        
    return [x,g1,ier]



if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()        

import numpy as np
import random
import numpy.linalg as la
import matplotlib.pyplot as plt

def driver1():
    N = 3
    Neval = 1000
    coefs = monomial(N  = 3)
    evals = np.zeros((N, 1))
    
    for i in range(Neval):
        x_i = -1 + (i - 1)*(2 / (Neval - 1))
        evals[i] = coefs[0] + coefs[1]*(x_i) + coefs[2]*(x_i)**2 + coefs[3]*(x_i)**3
        
    return evals
        

def monomial(N):
    N = 3
    vander = np.zeros((N + 1, N + 1))
    xvals = np.linspace(-1, 1, N + 1)
    
    f = lambda x: 1 / (1 + (10*x)**2)
    fvals = f(xvals)
    
    
    for i in range(N + 1):
        for j in range(N + 1):
            
            # populate column of ones
            if j == 0:
                vander[i][j] = 1
                
            # populate the rest of the matrix
            else:
                vander[i][j] = xvals[i]**j
    
    inverse = la.inv(vander)

    return inverse @ np.transpose(fvals)

monomial(N = 3)
    


#driver1()


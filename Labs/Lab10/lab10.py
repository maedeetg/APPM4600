import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math
from scipy.integrate import quad

def driver1():

    #  function you want to approximate
    f = lambda x: math.exp(x)

    # Interval of interest    
    a = -1
    b = 1
    # weight function    
    w = lambda x: 1.

    # order of approximation
    n = 2

    #  Number of points you want to sample in [a,b]
    N = 1000
    xeval = np.linspace(a,b,N+1)
    pval = np.zeros(N+1)

    for kk in range(N+1):
        pval[kk] = eval_legendre_expansion(f,a,b,w,n,xeval[kk])
      
    ''' create vector with exact values'''
    fex = np.zeros(N+1)
    for kk in range(N+1):
        fex[kk] = f(xeval[kk])
        
    plt.figure()    
    plt.plot(xeval,fex,'ro-', label= 'f(x)')
    plt.plot(xeval,pval,'bs--',label= 'Expansion') 
    plt.legend()
    plt.show()    
    
    err = abs(pval-fex)
    plt.semilogy(xeval,err,'ro--',label='error')
    plt.legend()
    plt.show()

def driver2():

    #  function you want to approximate
    f =  lambda x: 1 / (1 + x**2)

    # Interval of interest    
    a = -1
    b = 1
    # weight function    
    w = lambda x: 1.

    # order of approximation
    n = 2

    #  Number of points you want to sample in [a,b]
    N = 1000
    xeval = np.linspace(a,b,N+1)
    pval = np.zeros(N+1)

    for kk in range(N+1):
        pval[kk] = eval_legendre_expansion(f,a,b,w,n,xeval[kk])
      
    ''' create vector with exact values'''
    fex = np.zeros(N+1)
    for kk in range(N+1):
        fex[kk] = f(xeval[kk])
        
    plt.figure()    
    plt.plot(xeval,fex,'ro-', label= 'f(x)')
    plt.plot(xeval,pval,'bs--',label= 'Expansion') 
    plt.legend()
    plt.show()    
    
    err = abs(pval-fex)
    plt.semilogy(xeval,err,'ro--',label='error')
    plt.legend()
    plt.show()
    
def driver3():

    #  function you want to approximate
    f = lambda x: math.exp(x)

    # Interval of interest    
    a = -1
    b = 1
    # weight function    
    w = lambda x: 1 / (np.sqrt(1 - x**2))

    # order of approximation
    n = 2

    #  Number of points you want to sample in [a,b]
    N = 1000
    xeval = np.linspace(a,b,N+1)
    pval = np.zeros(N+1)

    for kk in range(N+1):
        pval[kk] = eval_legendre_expansion_cheb(f,a,b,w,n,xeval[kk])
      
    ''' create vector with exact values'''
    fex = np.zeros(N+1)
    for kk in range(N+1):
        fex[kk] = f(xeval[kk])
        
    plt.figure()    
    plt.plot(xeval,fex,'ro-', label= 'f(x)')
    plt.plot(xeval,pval,'bs--',label= 'Expansion') 
    plt.legend()
    plt.show()    
    
    err = abs(pval-fex)
    plt.semilogy(xeval,err,'ro--',label='error')
    plt.legend()
    plt.show()
    
def driver4():

    #  function you want to approximate
    f = lambda x: 1 / (1 + x**2)

    # Interval of interest    
    a = -1
    b = 1
    # weight function    
    w = lambda x: 1 / (np.sqrt(1 - x**2))

    # order of approximation
    n = 2

    #  Number of points you want to sample in [a,b]
    N = 1000
    xeval = np.linspace(a,b,N+1)
    pval = np.zeros(N+1)

    for kk in range(N+1):
        pval[kk] = eval_legendre_expansion_cheb(f,a,b,w,n,xeval[kk])
      
    ''' create vector with exact values'''
    fex = np.zeros(N+1)
    for kk in range(N+1):
        fex[kk] = f(xeval[kk])
        
    plt.figure()    
    plt.plot(xeval,fex,'ro-', label= 'f(x)')
    plt.plot(xeval,pval,'bs--',label= 'Expansion') 
    plt.legend()
    plt.show()    
    
    err = abs(pval-fex)
    plt.semilogy(xeval,err,'ro--',label='error')
    plt.legend()
    plt.show()

# Prelab
def eval_legendre(n, x):
    '''Inputs: order n, value x
       Output: vector p of length n + 1 whose
               entries are the values of the
               Legendre polynomials at x'''
    
    p = np.zeros(n + 1)
    
    phi_0 = 1
    phi_1 = x
    
    p[0] = phi_0
    p[1] = phi_1
    
    for i in range(1, n):
        p[i + 1] = (1 / (i + 1))*((2*i + 1)*x*p[i] - i*p[i - 1])
    
    return p

def eval_legendre_expansion(f,a,b,w,n,x): 
    pval = 0.0   

    for j in range(0, n+1):
        # make a function handle for evaluating phi_j(x)
        phi_j = lambda x: eval_legendre(n, x)[j]
        # make a function handle for evaluating phi_j^2(x)*w(x)
        phi_j_sq = lambda x: phi_j(x)**2*w(x)
        # use the quad function from scipy to evaluate normalizations
        norm_fac,err = quad(phi_j_sq, a, b)
        # make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
        func_j = lambda x: (phi_j(x)*f(x)*w(x)) / norm_fac
        # use the quad function from scipy to evaluate coeffs
        aj,err = quad(func_j, a, b)
        # accumulate into pval
        pval = pval+aj*phi_j(x) 

    return pval

def eval_legendre_cheb(n, x):
    '''Inputs: order n, value x
       Output: vector p of length n + 1 whose
               entries are the values of the
               Legendre polynomials at x'''
    
    t = np.zeros(n + 1)
    
    T_0 = 1
    T_1 = x
    
    t[0] = T_0
    t[1] = T_1
    
    for i in range(1, n):
        t[i + 1] = 2*x*t[i] - t[i - 1]
    
    return t

def eval_legendre_expansion_cheb(f,a,b,w,n,x): 
    pval = 0.0   

    for j in range(0, n+1):
        # make a function handle for evaluating phi_j(x)
        t_j = lambda x: eval_legendre_cheb(n, x)[j]
        # make a function handle for evaluating phi_j^2(x)*w(x)
        t_j_sq = lambda x: t_j(x)**2*w(x)
        # use the quad function from scipy to evaluate normalizations
        norm_fac,err = quad(t_j_sq, a, b)
        # make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
        func_j = lambda x: (t_j(x)*f(x)*w(x)) / norm_fac
        # use the quad function from scipy to evaluate coeffs
        aj,err = quad(func_j, a, b)
        # accumulate into pval
        pval = pval+aj*t_j(x) 

    return pval

# driver1()
# driver2()
# driver3()
driver4()
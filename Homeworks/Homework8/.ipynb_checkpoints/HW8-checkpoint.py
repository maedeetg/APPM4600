import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv 
from numpy.linalg import norm


# Problem 1.a)

def lagrange():
    a = -5
    b = 5
    f = lambda x: 1 / (1 + x**2)
    N = 20
    Neval = 1000
    xeval = np.linspace(a, b, Neval + 1)
    xint = np.linspace(a, b, N + 1)
    yint = f(xint)
    
    yeval_l = np.zeros(Neval + 1)
    
    for kk in range(Neval + 1):
        yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
    
    fex = f(xeval)

    plt.figure()    
    plt.plot(xeval,fex,'ro-', label = "f(x)")
    plt.plot(xeval,yeval_l,'bs--', label = "Lagrange")
    plt.title("Lagrange Polynomial for N = 20")
    plt.legend()
    plt.show()

    plt.figure() 
    err_l = abs(yeval_l-fex)
    plt.semilogy(xeval,err_l,'ro--',label='lagrange')
    plt.title("Error for Lagrange Polynomial for N = 20")
    plt.legend()
    plt.show()
    
# Problem 1.b)

def hermite():
    a = -5
    b = 5
    f = lambda x: 1 / (1 + x**2)
    fp = lambda x: -2*x/(1 + x**2)**2
    N = 20
    Neval = 1000
    xeval = np.linspace(a, b, Neval + 1)
    xint = np.linspace(a, b, N + 1)
    yint = f(xint)
    ypint = fp(xint)

    yevalH = np.zeros(Neval+1)
    
    for kk in range(Neval+1):
        yevalH[kk] = eval_hermite(xeval[kk],xint,yint,ypint,N)
        
    fex = f(xeval)
    
    plt.figure()
    plt.plot(xeval,fex,'ro-', label = 'f(x)') 
    plt.plot(xeval,yevalH,'c.--',label='Hermite')
    plt.semilogy()
    plt.title("Hermite Polynomial for N = 20")
    plt.legend()
    plt.show()
        
    errH = abs(yevalH-fex)
    plt.figure()
    plt.semilogy(xeval,errH,'bs--',label='Hermite')
    plt.title("Error for Hermite Polynomial for N = 20")
    plt.legend()
    plt.show() 
    
# Problem 1.c)   

def natural():
    f = lambda x: 1 / (1 + x**2)
    a = -5
    b = 5
    Nint = 5
    xint = np.linspace(a,b,Nint+1)
    yint = f(xint)
    Neval = 1000
    xeval =  np.linspace(xint[0], xint[Nint], Neval+1)
    
    (M,C,D) = create_natural_spline(yint,xint,Nint)
    yeval = eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D)
    
#    print('yeval = ', yeval)
    
    ''' evaluate f at the evaluation points'''
    fex = f(xeval)
        
    nerr = norm(fex-yeval)
    print('nerr = ', nerr)
    
    plt.figure()    
    plt.plot(xeval,fex,'ro-',label='f(x)')
    plt.plot(xeval,yeval,'bs--',label='Natural Spline') 
    plt.title("Natural Cubic Spline for N = 20")
    plt.legend()
    plt.show()
     
    err = abs(yeval-fex)
    plt.figure() 
    plt.semilogy(xeval,err,'ro--',label='Absolute error')
    plt.title("Absolute Error for Natural Cubic Spline for N = 20")
    plt.legend()
    plt.show()
    
# Problem 1.d)   

def clamped():
    f = lambda x: 1 / (1 + x**2)
    fp = lambda x: -2*x/(1 + x**2)**2
    a = -5
    b = 5
    Nint = 20
    xint = np.linspace(a,b,Nint+1)
    yint = f(xint)
    ypint = fp(xint)
    Neval = 1000
    xeval =  np.linspace(xint[0], xint[Nint], Neval+1)
    
    (M,C,D) = create_clamped_spline(yint,ypint,xint,Nint)
    yeval = eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D)
    
#    print('yeval = ', yeval)
    
    ''' evaluate f at the evaluation points'''
    fex = f(xeval)
        
    nerr = norm(fex-yeval)
    print('nerr = ', nerr)
    
    plt.figure()    
    plt.plot(xeval,fex,'ro-',label='f(x)')
    plt.plot(xeval,yeval,'bs--',label='Clamped Spline') 
    plt.title("Clamped Cubic Spline for N = 20")
    plt.legend()
    plt.show()
     
    err = abs(yeval-fex)
    plt.figure() 
    plt.semilogy(xeval,err,'ro--',label='Absolute error')
    plt.title("Absolute Error for Clamped Cubic Spline for N = 20")
    plt.legend()
    plt.show()
    
# Problem 2.a)

def lagrange_cheb():
    a = -5
    b = 5
    f = lambda x: 1 / (1 + x**2)
    N = 5
    Neval = 1000
    xeval = np.linspace(a, b, Neval + 1)
    xint = np.zeros((N + 1))
    
    for i in range(1, N + 1):
        xint[i - 1] = 5*np.cos(((2*i - 1)*np.pi) / (2*N))
        
    yint = f(xint)
    
    yeval_l = np.zeros(Neval + 1)
    
    for kk in range(Neval + 1):
        yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
    
    fex = f(xeval)

    plt.figure()    
    plt.plot(xeval,fex,'ro-', label = "f(x)")
    plt.plot(xeval,yeval_l,'bs--', label = "Lagrange")
    plt.title("Chebychev Lagrange Polynomial for N = 5")
    plt.legend()
    plt.show()

    plt.figure() 
    err_l = abs(yeval_l-fex)
    plt.semilogy(xeval,err_l,'ro--',label='lagrange')
    plt.title("Error for Chebychev Lagrange Polynomial for N = 5")
    plt.legend()
    plt.show()
    
# Problem 2.b)

def hermite_cheb():
    a = -5
    b = 5
    f = lambda x: 1 / (1 + x**2)
    fp = lambda x: -2*x/(1 + x**2)**2
    N = 20
    Neval = 1000
    xeval = np.linspace(a, b, Neval + 1)
    
    xint = np.zeros((N + 1))
    
    for i in range(1, N + 1):
        xint[i - 1] = 5*np.cos(((2*i - 1)*np.pi) / (2*N))
        
    yint = f(xint)
    ypint = fp(xint)

    yevalH = np.zeros(Neval+1)
    
    for kk in range(Neval+1):
        yevalH[kk] = eval_hermite(xeval[kk],xint,yint,ypint,N)
        
    fex = f(xeval)
    
    plt.figure()
    plt.plot(xeval,fex,'ro-', label = 'f(x)') 
    plt.plot(xeval,yevalH,'c.--',label='Hermite')
    plt.semilogy()
    plt.title("Chebychev Hermite Polynomial for N = 20")
    plt.legend()
    plt.show()
        
    errH = abs(yevalH-fex)
    plt.figure()
    plt.semilogy(xeval,errH,'bs--',label='Hermite')
    plt.title("Error for Chebychev Hermite Polynomial for N = 20")
    plt.legend()
    plt.show() 
    
# Problem 2.c)  

def natural_cheb():
    f = lambda x: 1 / (1 + x**2)
    a = -5
    b = 5
    Nint = 5
    
    xint = np.zeros((Nint + 1))
    
    for i in range(1, Nint + 2):
        xint[i - 1] = 5*np.cos(((2*i - 1)*np.pi) / (2*Nint))
        
    xint = xint[::-1]
    print(len(xint))
         
    yint = f(xint)
    Neval = 1000
    xeval =  np.linspace(xint[0], xint[Nint], Neval+1)
    
    (M,C,D) = create_natural_spline(yint,xint,Nint)
    yeval = eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D)
    
#    print('yeval = ', yeval)
    
    ''' evaluate f at the evaluation points'''
    fex = f(xeval)
        
    nerr = norm(fex-yeval)
    print('nerr = ', nerr)
    
    plt.figure()    
    plt.plot(xeval,fex,'ro-',label='f(x)')
    plt.plot(xeval,yeval,'bs--',label='Natural Spline') 
    plt.title("Natural Cubic Spline for N = 5")
    plt.legend()
    plt.show()
     
    err = abs(yeval-fex)
    plt.figure() 
    plt.semilogy(xeval,err,'ro--',label='Absolute error')
    plt.title("Absolute Error for Natural Cubic Spline for N = 5")
    plt.legend()
    plt.show()
         
# Problem 2.d)   

def clamped_cheb():
    f = lambda x: 1 / (1 + x**2)
    fp = lambda x: -2*x/(1 + x**2)**2
    a = -5
    b = 5
    Nint = 5
    
    xint = np.zeros((Nint + 1))
    
    for i in range(1, Nint + 2):
        xint[i - 1] = 5*np.cos(((2*i - 1)*np.pi) / (2*Nint))
        
    xint = xint[::-1]
    print(xint)
    
    yint = f(xint)
    ypint = fp(xint)
    Neval = 1000
    xeval =  np.linspace(xint[0], xint[Nint], Neval+1)
    
    (M,C,D) = create_clamped_spline(yint,ypint,xint,Nint)
    yeval = eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D)
    
#    print('yeval = ', yeval)
    
    ''' evaluate f at the evaluation points'''
    fex = f(xeval)
        
    nerr = norm(fex-yeval)
    print('nerr = ', nerr)
    
    plt.figure()    
    plt.plot(xeval,fex,'ro-',label='f(x)')
    plt.plot(xeval,yeval,'bs--',label='Clamped Spline') 
    plt.title("Chebychev Clamped Cubic Spline for N = 5")
    plt.legend()
    plt.show()
     
    err = abs(yeval-fex)
    plt.figure() 
    plt.semilogy(xeval,err,'ro--',label='Absolute error')
    plt.title("Absolute Error for Chebychev Clamped Cubic Spline for N = 5")
    plt.legend()
    plt.show()
    
def eval_lagrange(xeval, xint, yint, N):

    lj = np.ones(N + 1)
    
    for count in range(N + 1):
        for jj in range(N + 1):
            if (jj != count):
                lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N + 1):
        yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)

def eval_hermite(xeval,xint,yint,ypint,N):

    ''' Evaluate all Lagrange polynomials'''

    lj = np.ones(N+1)
    for count in range(N+1):
        for jj in range(N+1):
            if (jj != count):
                lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    ''' Construct the l_j'(x_j)'''
    lpj = np.zeros(N+1)
#    lpj2 = np.ones(N+1)
    for count in range(N+1):
        for jj in range(N+1):
            if (jj != count):
#              lpj2[count] = lpj2[count]*(xint[count] - xint[jj])
                lpj[count] = lpj[count]+ 1./(xint[count] - xint[jj])
              

    yeval = 0.
    
    for jj in range(N+1):
        Qj = (1.-2.*(xeval-xint[jj])*lpj[jj])*lj[jj]**2
        Rj = (xeval-xint[jj])*lj[jj]**2
#       if (jj == 0):
#         print(Qj)
         
#         print(Rj)
#         print(Qj)
#         print(xeval)
 #        return
        yeval = yeval + yint[jj]*Qj+ypint[jj]*Rj
       
    return(yeval)

def create_natural_spline(yint,xint,N):

#    create the right  hand side for the linear system
    b = np.zeros(N+1)
#  vector values
    h = np.zeros(N+1)  
    for i in range(1,N):
        hi = xint[i]-xint[i-1]
        hip = xint[i+1] - xint[i]
        
        if (hi == 0):
            hi += 10**(-10)
            
        elif (hip == 0):
            hi += 10**(-10)
            
        b[i] = ((yint[i+1]-yint[i])/hip) - ((yint[i]-yint[i-1])/hi)
        h[i-1] = hi
        h[i] = hip

#  create matrix so you can solve for the M values
# This is made by filling one row at a time 
    A = np.zeros((N+1,N+1))
    A[0][0] = 1.0
    
    for j in range(1,N):
        A[j][j-1] = h[j-1]/6
        A[j][j] = (h[j]+h[j-1])/3 
        A[j][j+1] = h[j]/6
        
    A[N][N] = 1

    Ainv = inv(A)
    
    M  = Ainv.dot(b)

#  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    
    for j in range(N):
        C[j] = (yint[j]/h[j])-(h[j]*M[j]/6)
        D[j] = (yint[j+1]/h[j])-(h[j]*M[j+1]/6)
        
    return(M,C,D)

def create_clamped_spline(yint,ypint,xint,N):

#    create the right  hand side for the linear system
    b = np.zeros(N+1)
#  vector values
    h = np.zeros(N+1) 
    h[0] = -ypint[0] + (yint[1] - yint[0]) / (xint[1] - xint[0])
    h[N] = -ypint[N] + (yint[N] - yint[N - 1]) / (xint[N] - xint[N - 1])
    for i in range(1, N):
        hi = xint[i]-xint[i-1]
        hip = xint[i+1] - xint[i]
        
        if (hi == 0):
            hi += 10**(-10)
            
        elif (hip == 0):
            hi += 10**(-10)
            
        b[i] = (yint[i+1]-yint[i])/hip - (yint[i]-yint[i-1])/hi
        h[i-1] = hi
        h[i] = hip

#  create matrix so you can solve for the M values
# This is made by filling one row at a time 
    A = np.zeros((N+1,N+1))
    A[0][0] = ((xint[1] + 10**(-10)) - xint[0]) / 3
    A[0][1] = ((xint[1] + 10**(-10)) - xint[0]) / 6
    A[N][N - 1] = ((xint[N] + 10**(-10))- xint[N - 1]) / 6
    A[N][N] = ((xint[N] + 10**(-10)) - xint[N - 1]) / 3
    
    for j in range(1,N):
        A[j][j-1] = h[j-1]/6
        A[j][j] = (h[j]+h[j-1])/3 
        A[j][j+1] = h[j]/6
    
    Ainv = inv(A)
    
    M  = Ainv.dot(b)

#  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    
    for j in range(N):
        C[j] = yint[j]/h[j]-h[j]*M[j]/6
        D[j] = yint[j+1]/h[j]-h[j]*M[j+1]/6
        
    return(M,C,D)

def eval_local_spline(xeval,xi,xip,Mi,Mip,C,D):
# Evaluates the local spline as defined in class
# xip = x_{i+1}; xi = x_i
# Mip = M_{i+1}; Mi = M_i

    hi = xip-xi
    yeval = (Mi*(xip-xeval)**3 +(xeval-xi)**3*Mip)/(6*hi) + C*(xip-xeval) + D*(xeval-xi)
    return yeval 

def  eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D):
    
    yeval = np.zeros(Neval+1)
    
    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j]
        btmp= xint[j+1]
        
#   find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]

# evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,M[j],M[j+1],C[j],D[j])
#        print('yloc = ', yloc)
#   copy into yeval
        yeval[ind] = yloc

    return(yeval)

clamped_cheb()
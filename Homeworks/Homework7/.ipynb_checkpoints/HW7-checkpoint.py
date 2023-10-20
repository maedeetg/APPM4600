import numpy as np
import random
import numpy.linalg as la
import matplotlib.pyplot as plt

# Problem 1 a

def driver1():
    N = 5
    xvals = np.zeros(N)

    for i in range(1, N + 1):
        xvals[i - 1] = -1 + (i - 1)*(2 / (N - 1))

    f = lambda x: 1 / (1 + (10*x)**2)
    fvals = f(xvals)
    coefs = monomial(N, xvals, fvals)

    Neval = 1000
    xevals = np.linspace(-1, 1, Neval)
    yevals = np.zeros(Neval)
    
    for i in range(Neval):
        x = xevals[i]
        
        for j in range(len(coefs)):
            yevals[i] += coefs[j]*(x**j)
            
    plt.plot(xevals, f(xevals), 'ro-')
    plt.plot(xevals, yevals, 'bo-')
    plt.title("N = 5 Plot")
    plt.show()
    
def driver2():
    N = 10
    xvals = np.zeros(N)

    for i in range(1, N + 1):
        xvals[i - 1] = -1 + (i - 1)*(2 / (N - 1))

    f = lambda x: 1 / (1 + (10*x)**2)
    fvals = f(xvals)
    coefs = monomial(N, xvals, fvals)

    Neval = 1000
    xevals = np.linspace(-1, 1, Neval)
    yevals = np.zeros(Neval)
    
    for i in range(Neval):
        x = xevals[i]
        
        for j in range(len(coefs)):
            yevals[i] += coefs[j]*(x**j)
            
    plt.plot(xevals, f(xevals), 'ro-')
    plt.plot(xevals, yevals, 'bo-')
    plt.title("N = 10 Plot")
    plt.show()
    
def driver3():
    N = 15
    xvals = np.zeros(N)

    for i in range(1, N + 1):
        xvals[i - 1] = -1 + (i - 1)*(2 / (N - 1))

    f = lambda x: 1 / (1 + (10*x)**2)
    fvals = f(xvals)
    coefs = monomial(N, xvals, fvals)

    Neval = 1000
    xevals = np.linspace(-1, 1, Neval)
    yevals = np.zeros(Neval)
    
    for i in range(Neval):
        x = xevals[i]
        
        for j in range(len(coefs)):
            yevals[i] += coefs[j]*(x**j)
            
    plt.plot(xevals, f(xevals), 'ro-')
    plt.plot(xevals, yevals, 'bo-')
    plt.title("N = 15 Plot")
    plt.show()
    
def driver4():
    N = 17
    xvals = np.zeros(N)

    for i in range(1, N + 1):
        xvals[i - 1] = -1 + (i - 1)*(2 / (N - 1))

    f = lambda x: 1 / (1 + (10*x)**2)
    fvals = f(xvals)
    coefs = monomial(N, xvals, fvals)

    Neval = 1000
    xevals = np.linspace(-1, 1, Neval)
    yevals = np.zeros(Neval)
    
    for i in range(Neval):
        x = xevals[i]
        
        for j in range(len(coefs)):
            yevals[i] += coefs[j]*(x**j)
            
    plt.plot(xevals, f(xevals), 'ro-')
    plt.plot(xevals, yevals, 'bo-')
    plt.title("N = 17 Plot")
    plt.show()
        
# Problem 2, i have a bug that I cannot figure out

def driver5():
    N = 5
    xvals = np.zeros(N)

    for i in range(1, N + 1):
        xvals[i - 1] = -1 + (i - 1)*(2 / (N - 1))

    print(xvals)
    f = lambda x: 1 / (1 + (10*x)**2)
    fvals = f(xvals)
    coefs = np.ones(N)
    yvals = np.zeros(N)
    w = np.ones(N)
    
    for i in range(N):
        for j in range(N):
            if i != j:
                w[i] *= (xvals[i] - xvals[j])**(-1)
    
    Neval = 1000
    xevals = np.linspace(-1, 1, Neval)
    yevals = np.zeros(Neval)
    
    # for i in range(N):
    #     for j in range(Neval):
    #         print('x', xevals[j], 'xj', xvals[i])
    #         coefs[i] *= xevals[j] - xvals[i]
    
    for j in range(Neval):
        for i in range(N):
            # print('x', xevals[j], 'i', i)
            #print('x', xevals[j], 'xj', xvals[i])
            coef = eval_coefs(i, xevals[j], xvals)
            w = eval_w(i, j, xevals, xvals)
            print(w)
            # print('c', coef)
            yevals[j] = coef*(w / (xevals[j] - xvals[i]))*f(xvals[i])

    #print(yevals)
    plt.plot(xevals, f(xevals), 'ro-')
    plt.plot(xevals, yevals, 'bo-')
    plt.show()

def eval_coefs(N, x, xvals):
    coefs = 1
    
    for i in range(N):
        coefs *= x - xvals[i]
        
    return coefs

def eval_w(N, j, xevals,xvals):
    w = 1
    x_j = xevals[j]
    
    for i in range(N):
        if i != j:
            w *= (x_j - xvals[i])**(-1)
        
    
    # for i in range(N):
    #     for j in range(N):
    #         if i != j:
    #             w *= (xvals[i] - xvals[j])**(-1)
            
    return w
    
            
#     plt.plot(xevals, f(xevals), 'ro-')
#     plt.plot(xevals, yevals, 'bo-')
#     plt.title("N = 17 Plot")
#     plt.show()
    
def monomial(N, xvals, fvals):
    vander = np.zeros([N, N])
    
    for i in range(N):
        for j in range(N):
            vander[i][j] = xvals[i]**j
    
    inverse = la.inv(vander)
    
    return inverse.dot(fvals)

# driver1()
# driver2()
# driver3()
# driver4()
driver5()
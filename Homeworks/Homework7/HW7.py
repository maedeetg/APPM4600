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
        
# Problem 2

def driver5():
    N = 5
    xvals = np.zeros(N)

    for i in range(1, N + 1):
        xvals[i - 1] = -1 + (i - 1)*(2 / (N - 1))

    f = lambda x: 1 / (1 + (10*x)**2)
    fvals = f(xvals)
    coefs = np.ones(N)
    yvals = np.zeros(N)
    
    Neval = 1000
    xevals = np.linspace(-1, 1, Neval)
    yevals = np.zeros(Neval)
    
    for i in range(Neval):
        for j in range(N):
            w = eval_w(N, j, xvals)
            yevals[i] += (w / (xevals[i] - xvals[j]))*f(xvals[j])
            
        coef = eval_coefs(N, xevals[i], xvals)
        yevals[i] = coef*yevals[i]
        
    plt.plot(xevals, f(xevals), 'ro-')
    plt.plot(xevals, yevals, 'bo-')
    plt.title("N = 5 Barycentric Plot")
    plt.show()
    
def driver6():
    N = 8
    xvals = np.zeros(N)

    for i in range(1, N + 1):
        xvals[i - 1] = -1 + (i - 1)*(2 / (N - 1))

    f = lambda x: 1 / (1 + (10*x)**2)
    fvals = f(xvals)
    coefs = np.ones(N)
    yvals = np.zeros(N)
    
    Neval = 1000
    xevals = np.linspace(-1, 1, Neval)
    yevals = np.zeros(Neval)
    
    for i in range(Neval):
        for j in range(N):
            w = eval_w(N, j, xvals)
            yevals[i] += (w / (xevals[i] - xvals[j]))*f(xvals[j])
            
        coef = eval_coefs(N, xevals[i], xvals)
        yevals[i] = coef*yevals[i]
        
    plt.plot(xevals, f(xevals), 'ro-')
    plt.plot(xevals, yevals, 'bo-')
    plt.title("N = 8 Barycentric Plot")
    plt.show()
    
def driver7():
    N = 10
    xvals = np.zeros(N)

    for i in range(1, N + 1):
        xvals[i - 1] = -1 + (i - 1)*(2 / (N - 1))

    f = lambda x: 1 / (1 + (10*x)**2)
    fvals = f(xvals)
    coefs = np.ones(N)
    yvals = np.zeros(N)
    
    Neval = 1000
    xevals = np.linspace(-1, 1, Neval)
    yevals = np.zeros(Neval)
    
    for i in range(Neval):
        for j in range(N):
            w = eval_w(N, j, xvals)
            yevals[i] += (w / (xevals[i] - xvals[j]))*f(xvals[j])
            
        coef = eval_coefs(N, xevals[i], xvals)
        yevals[i] = coef*yevals[i]
        
    plt.plot(xevals, f(xevals), 'ro-')
    plt.plot(xevals, yevals, 'bo-')
    plt.title("N = 10 Barycentric Plot")
    plt.show()

def driver8():
    N = 15
    xvals = np.zeros(N)

    for i in range(1, N + 1):
        xvals[i - 1] = -1 + (i - 1)*(2 / (N - 1))

    f = lambda x: 1 / (1 + (10*x)**2)
    fvals = f(xvals)
    coefs = np.ones(N)
    yvals = np.zeros(N)
    
    Neval = 1000
    xevals = np.linspace(-1, 1, Neval)
    yevals = np.zeros(Neval)
    
    for i in range(Neval):
        for j in range(N):
            w = eval_w(N, j, xvals)
            yevals[i] += (w / (xevals[i] - xvals[j]))*f(xvals[j])
            
        coef = eval_coefs(N, xevals[i], xvals)
        yevals[i] = coef*yevals[i]
        
    plt.plot(xevals, f(xevals), 'ro-')
    plt.plot(xevals, yevals, 'bo-')
    plt.title("N = 15 Barycentric Plot")
    plt.show()

def driver9():
    N = 20
    xvals = np.zeros(N)

    for i in range(1, N + 1):
        xvals[i - 1] = -1 + (i - 1)*(2 / (N - 1))

    f = lambda x: 1 / (1 + (10*x)**2)
    fvals = f(xvals)
    coefs = np.ones(N)
    yvals = np.zeros(N)
    
    Neval = 1000
    xevals = np.linspace(-1, 1, Neval)
    yevals = np.zeros(Neval)
    
    for i in range(Neval):
        for j in range(N):
            w = eval_w(N, j, xvals)
            yevals[i] += (w / (xevals[i] - xvals[j]))*f(xvals[j])
            
        coef = eval_coefs(N, xevals[i], xvals)
        yevals[i] = coef*yevals[i]
            
    plt.plot(xevals, f(xevals), 'ro-')
    plt.plot(xevals, yevals, 'bo-')
    plt.title("N = 20 Barycentric Plot")
    plt.show()
    
# Problem 3

def driver10():
    N = 5
    xvals = np.zeros(N)

    for i in range(1, N + 1):
        # np.cos(((2*i - 1)*np.pi) / 2*N)
        xvals[i - 1] = np.cos(((2*i - 1)*np.pi) / (2*N))

    f = lambda x: 1 / (1 + (10*x)**2)
    fvals = f(xvals)
    coefs = np.ones(N)
    yvals = np.zeros(N)
    
    Neval = 1000
    xevals = np.linspace(-1, 1, Neval)
    yevals = np.zeros(Neval)
    
    for i in range(Neval):
        for j in range(N):
            w = eval_w(N, j, xvals)
            yevals[i] += (w / (xevals[i] - xvals[j]))*f(xvals[j])
            
        coef = eval_coefs(N, xevals[i], xvals)
        yevals[i] = coef*yevals[i]
            
    plt.plot(xevals, f(xevals), 'ro-')
    plt.plot(xevals, yevals, 'bo-')
    plt.title("N = 5 Barycentric Plot")
    plt.show()
    
def driver11():
    N = 10
    xvals = np.zeros(N)

    for i in range(1, N + 1):
        # np.cos(((2*i - 1)*np.pi) / 2*N)
        xvals[i - 1] = np.cos(((2*i - 1)*np.pi) / (2*N))

    f = lambda x: 1 / (1 + (10*x)**2)
    fvals = f(xvals)
    coefs = np.ones(N)
    yvals = np.zeros(N)
    
    Neval = 1000
    xevals = np.linspace(-1, 1, Neval)
    yevals = np.zeros(Neval)
    
    for i in range(Neval):
        for j in range(N):
            w = eval_w(N, j, xvals)
            yevals[i] += (w / (xevals[i] - xvals[j]))*f(xvals[j])
            
        coef = eval_coefs(N, xevals[i], xvals)
        yevals[i] = coef*yevals[i]
            
    plt.plot(xevals, f(xevals), 'ro-')
    plt.plot(xevals, yevals, 'bo-')
    plt.title("N = 10 Barycentric Plot")
    plt.show()
    
def driver12():
    N = 15
    xvals = np.zeros(N)

    for i in range(1, N + 1):
        # np.cos(((2*i - 1)*np.pi) / 2*N)
        xvals[i - 1] = np.cos(((2*i - 1)*np.pi) / (2*N))

    f = lambda x: 1 / (1 + (10*x)**2)
    fvals = f(xvals)
    coefs = np.ones(N)
    yvals = np.zeros(N)
    
    Neval = 1000
    xevals = np.linspace(-1, 1, Neval)
    yevals = np.zeros(Neval)
    
    for i in range(Neval):
        for j in range(N):
            w = eval_w(N, j, xvals)
            yevals[i] += (w / (xevals[i] - xvals[j]))*f(xvals[j])
            
        coef = eval_coefs(N, xevals[i], xvals)
        yevals[i] = coef*yevals[i]
            
    plt.plot(xevals, f(xevals), 'ro-')
    plt.plot(xevals, yevals, 'bo-')
    plt.title("N = 15 Barycentric Plot")
    plt.show()
    
def driver13():
    N = 20
    xvals = np.zeros(N)

    for i in range(1, N + 1):
        # np.cos(((2*i - 1)*np.pi) / 2*N)
        xvals[i - 1] = np.cos(((2*i - 1)*np.pi) / (2*N))

    f = lambda x: 1 / (1 + (10*x)**2)
    fvals = f(xvals)
    coefs = np.ones(N)
    yvals = np.zeros(N)
    
    Neval = 1000
    xevals = np.linspace(-1, 1, Neval)
    yevals = np.zeros(Neval)
    
    for i in range(Neval):
        for j in range(N):
            w = eval_w(N, j, xvals)
            yevals[i] += (w / (xevals[i] - xvals[j]))*f(xvals[j])
            
        coef = eval_coefs(N, xevals[i], xvals)
        yevals[i] = coef*yevals[i]
            
    plt.plot(xevals, f(xevals), 'ro-')
    plt.plot(xevals, yevals, 'bo-')
    plt.title("N = 20 Barycentric Plot")
    plt.show()
    
def driver14():
    N = 30
    xvals = np.zeros(N)

    for i in range(1, N + 1):
        # np.cos(((2*i - 1)*np.pi) / 2*N)
        xvals[i - 1] = np.cos(((2*i - 1)*np.pi) / (2*N))

    f = lambda x: 1 / (1 + (10*x)**2)
    fvals = f(xvals)
    coefs = np.ones(N)
    yvals = np.zeros(N)
    
    Neval = 1000
    xevals = np.linspace(-1, 1, Neval)
    yevals = np.zeros(Neval)
    
    for i in range(Neval):
        for j in range(N):
            w = eval_w(N, j, xvals)
            yevals[i] += (w / (xevals[i] - xvals[j]))*f(xvals[j])
            
        coef = eval_coefs(N, xevals[i], xvals)
        yevals[i] = coef*yevals[i]
            
    plt.plot(xevals, f(xevals), 'ro-')
    plt.plot(xevals, yevals, 'bo-')
    plt.title("N = 30 Barycentric Plot")
    plt.show()

def eval_coefs(N, x, xvals):
    coefs = 1
    
    for i in range(N):
        coefs *= x - xvals[i]
        
    return coefs

def eval_w(N, j, xvals):
    w = 1
    x_j = xvals[j]
    
    for i in range(N):
        if i != j:
            w *= (x_j - xvals[i])**(-1)
            
    return w

def monomial(N, xvals, fvals):
    vander = np.zeros([N, N])
    
    for i in range(N):
        for j in range(N):
            vander[i][j] = xvals[i]**j
    
    inverse = la.inv(vander)
    
    return inverse.dot(fvals)

driver1()
driver2()
driver3()
driver4()
driver5()
driver6()
driver7()
driver8()
driver9()
driver10()
driver11()
driver12()
driver13()
driver14()
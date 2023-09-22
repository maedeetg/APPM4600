# import libraries
import numpy as np
import math

def driver():

# functions

    g = lambda x: (10 / (x + 4))**(1/2)
    x0 = 1.5
    tol = 10e-10

    seq1 = fixedpt(g, x0, tol, 20)
    print('The number of iterations with fixed point is:', len(seq1))
    aitkens(seq1, tol, Nmax = len(seq1))
    
    f = lambda x: x - ((x**5 - 7)/12)
    x1 = 1.0
    tol = 10e-10
    
    seq2 = fixedpt(f, x1, tol, 2000)
    print('The number of iterations with fixed point is:', len(seq2))
    aitkens(seq2, tol, Nmax = len(seq2))
    
    return

# define routines

def fixedpt(f, x0, tol, Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    x = []
    
    while (count < Nmax):
        
        count = count +1
        x1 = f(x0)
        # x.append(x1)
        
        if (abs(x1 - x0) < tol):
            xstar = x1
            ier = 0
            x.append(xstar)
            return x
        
        x0 = x1
        x.append(x0)

    xstar = x1
    ier = 1
    x.append(xstar)
    
    return x

def aitkens(seq, tol, Nmax):
    
    aitkens = []
    
    p0 = seq[0]
    p1 = seq[1]
    p2 = seq[2]
    
    aitkens.append(p0 - ((p1 - p2)**2 / (p2 - 2*p1 + p0)))
    
    n = 0
    
    while (n < Nmax):
        n = n + 1
        
        aitkens.append((seq[n] - ((seq[n + 1] - seq[n])**2 / (seq[n + 2] - 2*seq[n + 1] + seq[n]))))
        
        if (abs(aitkens[n - 1] - aitkens[n]) < tol):
            print('The number of iterations with Aitkens is:', len(aitkens))
            
            return aitkens
     
    print('The number of iterations with Aitkens is:', len(aitkens))
    
    return

driver()
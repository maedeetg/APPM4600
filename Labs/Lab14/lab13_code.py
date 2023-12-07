import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
import time


def LU():
    N = 5000
    
    start1 = time.time()
    b = np.random.rand(N,1)
    A = np.random.rand(N,N)
    LU, P = scila.lu_factor(A)
    end1 = time.time()
    
    start2 = time.time()
    x = scila.lu_solve((LU, P), b)
    end2 = time.time()
    print("LU Construction Time: ", end1 - start1, "LU Solve Time: ", end2 - start2)


    test = np.matmul(A,x)
    r = la.norm(test-b)

    print(r)

    N = 10
    M = 5
    A = create_rect(N,M)     
    b = np.random.rand(N,1)
    

def QR():
    N = 100

    b = np.random.rand(N,1)
    A = np.random.rand(N,N)
    Q, R = scila.qr(A)
    
    Qb = np.dot(Q.T, b)
    x_qr = scila.lstsq(R, Qb)[0]
    test_qr = np.matmul(A, x_qr)
    r_qr = la.norm(test_qr - b)
    print(r_qr)

    N = 10
    M = 5
    A = create_rect(N,M)     
    b = np.random.rand(N,1)
     
def create_rect(N,M):
    a = np.linspace(1,10,M)
    d = 10**(-a)

    D2 = np.zeros((N,M))
    for j in range(0,M):
        D2[j,j] = d[j]

    A = np.random.rand(N,N)
    Q1, R = la.qr(A)
    test = np.matmul(Q1,R)
    A =    np.random.rand(M,M)
    Q2,R = la.qr(A)
    test = np.matmul(Q2,R)

    B = np.matmul(Q1,D2)
    B = np.matmul(B,Q2)
    return B     
          
LU()       

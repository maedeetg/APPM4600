import numpy as np
import numpy.linalg as la
import math
import time

def driver():
    # question 1
    n = 100
    a = np.linspace(0, np.pi, n)
    b = 0 * a

    # compute the dot prodcut between two vectors a and b that
    # are both size n
    dp = dotProduct(a, b, n)
    print('The dot product is: ', dp)
    
    # question 2
    # 2 by 2 matrix vector multipliation
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[2], [1]])
    C = matrix_mult(A, B)
    print('The matrix vector multiplication of A and B is: ', C)
    
    # 6 by 6 matrix vector multiplication
    A2 = np.array([[1, 2, 3, 4, 5, 6],
                   [7, 8, 9, 10, 11, 12],
                   [13, 14, 15, 16, 17, 18],
                   [19, 20, 21, 22, 23, 24],
                   [25, 26, 27, 28, 29, 30],
                   [31, 32, 33, 34, 35, 36]])
    B2 = np.array([[6], [5], [4], [3], [2], [1]])
    C2 = matrix_mult(A2, B2) # C is the resulting matrix after multiplying A and B
    print('The matrix vector multiplication of A2 and B2 is: ', C2)
    
    # question 3
    # for the sake of testing runtime, I made larger vectors and matrices
    n2 = 10000
    a2 = np.linspace(0, np.pi, n2)
    b2 = 0 * a2
    
    # the following code will output the runtime of each method
    # in order to compare which methods are faster
    
    # run time for my dot product
    start_time1 = time.time()
    dp2 = dotProduct(a2, b2, n2)
    end_time1 = time.time()
    
    # run time for matrix_mult
    A3 = np.ones((30, 30))
    start_time2 = time.time()
    C3 = matrix_mult(A3, A3)
    end_time2 = time.time()
    
    # run time for numpy dot product
    start_time3 = time.time()
    new_dot = np.dot(a2, b2)
    end_time3 = time.time()
    
    start_time4 = time.time()
    new_mult = np.matmul(A3, A3)
    end_time4 = time.time()
    
    # Printing runtimes of different methods. For testing, I made larger vectors and matrices
    print(f'My Dot Prod Runtime:{end_time1 - start_time1:10f}, My Matrix Mult Runtime:{end_time2 - start_time2:10f}, Numpy Dot Prod Runtime: {end_time3 - start_time3:10f}, Numpy Matrix Mult Runtime:{end_time4 - start_time4:10f}')
    
    # The numpy methods are faster when the size of the vectors and matrices increase
    
    # verify both methods do the same thing
    print(f'Are dp2 and new_dot equal? {dp2 == new_dot}, Are C3 and new_mult equal? {C3 == new_mult}')
    
    return
    
    
def dotProduct(x, y, n):
    '''This function will compute the
       dot product of two vectors x, y
       of size n'''
    
    dp = 0. # initilize dot product
    
    # this for loop will go through all n elements in the vectors
    # and compute their product, and then update dp
    for j in range(n):
        dp = dp + x[j]*y[j]

    return dp 

def matrix_mult(A, B):
    '''This function will compute a matrix
       vector multiplication of two matrices
       A and B'''
    
    m1, n1 = A.shape # m1 is number of rows, n1 is number of columns
    m2, n2 = B.shape # m2 is number of rows, n2 is number of columns
    
    C = np.zeros((m1, n2)) # intilize final matrix, size is n1 by m2
    
    # this nested for loop will go through each column and row and compute the dot product, then update the ith, jth element of C
    for i in range(0, n1):
        for j in range(0, n2):
            C[i][j] = dotProduct(A[i, :], B[:, j], m2)
            
    return C

driver()
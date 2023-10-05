import numpy as np
import numpy.linalg as nplin

# Problem 1a

# function evaluaters

def driver1():
    # functions
    f = lambda x, y: 3*x**2 - y**2
    g = lambda x, y: 3*x*y**2 - x**3 - 1
    
    # initial values
    x0 = 1
    y0 = 1
    
    b = np.array([[x0],[y0]]) # (2x1)
    
    # Jacobian matrix (2x2)
    J = np.array([[1/6, 1/18], [0, 1/6]])
    
    vals = []
    
    for i in range(30):
        x_n = b[0][0]
        y_n = b[1][0]
        
        # function vector
        f_vec = np.array([[f(x_n, y_n)], [g(x_n, y_n)]])
        
        # x_n+1 and y_n+1
        new = b - J@f_vec
        
        vals.append(new)
        b = new
        
        print(f'Iteration: {i + 1}, x_(n+1): {vals[i][0]}, y_(n+1): {vals[i][1]}')
        print('')
        
    return

# Problem 1c
    
def driver2():
    f = lambda x, y: 3*x**2 - y**2
    g = lambda x, y: 3*x*y**2 - x**3 - 1
    J = jacobian()
    x0 = 1
    y0 = 1
    tol = 10**(-10)
    Nmax = 100
    
    [ier, it, pstar] = newton2d(f, g, J, x0, y0, tol, Nmax)
    print("The estimated solution is:", pstar)
    print("The error message reads:", ier)
    print("The number of iterations:", it)
    
    return

# Problem 3b

def driver3():
    f = lambda x, y, z: x**2 + 4*y**2 + 4*z**2 - 16
    fx = lambda x: 2*x
    fy = lambda y: 8*y
    fz = lambda z: 8*z
    
    x0 = 1
    y0 = 1
    z0 = 1
    prev = np.array([[x0], [y0], [z0]])
    
    tol = 10**(-10)
    Nmax = 50
    
    for n in range(Nmax):
        x_n = prev[0][0]
        y_n = prev[1][0]
        z_n = prev[2][0]
        
        new = prev - (f(x_n, y_n, z_n) / ((fx(x_n))**2 + (fy(y_n))**2 + (fz(z_n))**2))*np.array([[fx(x_n)], [fy(y_n)], [fz(z_n)]])
        
        if (nplin.norm(new - prev, 2) < tol):
            prev = new
            count = n
            
            print("The number of iterations needed:", count)
            print("The estimated point is:", prev)
            return
        
        prev = new
     
    res = new
    print("The number of iterations needed:", count)
    print("The estimated point is:", res)
    return

def jacobian():
    f = lambda x, y: 3*x**2 - y**2
    g = lambda x, y: 3*x*y**2 - x**3 - 1
    
    df_x = lambda x, y: 6*x
    df_y = lambda x, y: -2*y
    dg_x = lambda x, y: 3*y**2 - 3*x**2
    dg_y = lambda x, y: 6*x*y
    
    return [df_x, df_y, dg_x, dg_y]
    
def jacobian_eval(J, x, y):
    [df_x, df_y, dg_x, dg_y] = J
    
    J = np.array([[df_x(x, y), -df_y(x, y)], [-dg_x(x, y), dg_y(x, y)]])
    det_J = nplin.det(J)
    
    J = (1 / int(det_J)) * J
    
    return J
         
def newton2d(f, g, J, x0, y0, tol, Nmax):
    # initial values
    x0 = 1
    y0 = 1
    
    b = np.array([[x0],[y0]]) # (2x1)
    
    vals = []
    
    for i in range(Nmax):
        x_n = b[0][0]
        y_n = b[1][0]
        
        # function vector
        f_vec = np.array([[f(x_n, y_n)], [g(x_n, y_n)]])
        
        # Jacobian matrix (2x2)
        Jac = jacobian_eval(J, x_n, y_n)
        
        # x_n+1 and y_n+1
        new = b - Jac@f_vec
        
        if (nplin.norm(new - b, 2) < tol):
            vals.append(new)
            pstar = new
            ier = 0
            it = i
            
            return [ier, it, pstar]
        
        b = new
    
    pstar = new
    vals.append(new)
    ier = 0
    it = Nmax
    
    return [ier, it, pstar]

driver1()
driver2()
driver3()
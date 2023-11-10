import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# Question 1 error

def driver1():
    a = lambda x: (x + (-7/60)*x**3) / (1 + (1/20)*x**2)
    b = lambda x: x / (1 + (1/6)*x**2 + (7/360)*x**4)
    c = a
    
    sixth = lambda x: 1 + x + (1/2)*x**2 + (1/6)*x**3 + (1/24)*x**4 + (1/120)*x**5 + (1/720)*x**6
    
    xvals = np.linspace(0, 5, 100)
    err1 = abs(sixth(xvals) - a(xvals))
    err2 = abs(sixth(xvals) - b(xvals))
    err3 = abs(sixth(xvals) - c(xvals))
    
    plt.semilogy(xvals, err1, label = "1a", linewidth = 2)
    plt.semilogy(xvals, err2, label = "1b", linewidth = 2)
    plt.semilogy(xvals, err3, label = "1c", linewidth = 2)
    plt.title("Pade Approximation Errors for Question 1 HW10")
    plt.legend()
    plt.show()

def driver2():
    f = lambda s: 1 / (1 + s**2)
    a = -5
    b = 5
    N = 1000
    
    trap_approx = comp_trap(a, b, N, f)
    simp_approx = comp_simp(a, b, N, f)
    
    print("Composite Trapezoidal Approximation: ", trap_approx)
    print("Composite Simpson Approximation: ", simp_approx)
    print("")
    
def driver3():
    d2f = lambda s: -2*((-3*s**2 + 1) / (1 + s**2)**3)
    d4f = lambda s: 24*(5*s**4 - 10*s**2 + 1) / (s**2 + 1)**5
    a = -5
    b = 5
    n1 = comp_trap_error(a, b, d2f)
    n2 = comp_simp_error(a, b, d4f)
    
    print("Composite Trapezoidal n-value: ", n1)
    print("Composite Simpson n-value: ", n2)
    print("")
    
def driver4():
    f = lambda s: 1 / (1 + s**2)
    d2f = lambda s: -2*((-3*s**2 + 1) / (1 + s**2)**3)
    d4f = lambda s: 24*(5*s**4 - 10*s**2 + 1) / (s**2 + 1)**5
    a = -5
    b = 5
    n1 = comp_trap_error(a, b, d2f)
    n2 = comp_simp_error(a, b, d4f)
    
    x = compare(a, b, n1, n2, f)
    
    (pred_t, pred_s, exact1, exact2, nval1, nval2) = compare(a, b, n1, n2, f)
    
    print("Exact with tolerance 10e-4 : ", exact2[0])
    print("Neval with tolerane 10e-4: ", nval2)
    print("")
    print("Exact with tolerance 10e-6: ", exact1[0])
    print("Neval with tolerane 10e-6: ", nval1)
    print("")
    print("Predicted Trapezoidal: ", pred_t)
    print("Predicted Simpson: ", pred_s)
    print("")
    
def comp_trap(a, b, N, f):
    h = (b - a) / N
    xvals = np.linspace(a, b, N + 1)
    
    middle = 0
    
    for i in range(1, N):
        middle += f(xvals[i])
        
    final = (h/2) * (f(a) + 2*middle + f(b))
    
    return final

def comp_simp(a, b, N, f):
    h = (b - a) / N
    xvals = np.linspace(a, b, N + 1)
    
    middle1 = 0
    middle2 = 0
    
    for i in range(1, N//2):
        middle1 += f(xvals[2*i])
        
    for i in range(1, N//2 + 1):
        middle2 += f(xvals[2*i - 1])
        
    final = (h/3)*(f(a) + 2*middle1 + 4*middle2 + f(b))
    
    return final
    
def comp_trap_error(a, b, d2f):
    xvals_temp = np.linspace(a, b, 10000)
    yvals_temp = np.abs(d2f(xvals_temp))
    max_val = max(yvals_temp)
    
    n_guess = np.sqrt(np.abs((max_val*(b - a)**3) / (12 * 10**(-4))))

    return math.ceil(n_guess)

def comp_simp_error(a, b, d4f):
    xvals_temp = np.linspace(a, b, 10000)
    yvals_temp = np.abs(d4f(xvals_temp))
    max_val = max(yvals_temp)

    n_guess = pow(np.abs((max_val*(b - a)**5) / (180 * 10**(-4))), 1/4)
    
    return math.ceil(n_guess)

def compare(a, b, n1, n2, f):
    pred_t = comp_trap(a, b, n1, f)
    pred_s = comp_simp(a, b, n2, f)
    exact1 = integrate.quad(f, a, b, full_output = 1)
    exact2 = integrate.quad(f, a, b, epsabs = 10**(-4), full_output = 1)
    
    nval1 = exact1[2]['neval']
    nval2 = exact2[2]['neval']
    return(pred_t, pred_s, exact1, exact2, nval1, nval2)
    
driver1()
driver2()
driver3()
driver4()
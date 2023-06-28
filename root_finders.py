"""
MATH2019 CW1 rootfinders module

@author: LEE YEN MAY
"""

import numpy as np
import matplotlib.pyplot as plt

def bisection(f,a,b,Nmax):
    
    """
    Bisection Method: Returns a numpy array of the 
    sequence of approximations obtained by the bisection method.
    
    Parameters
    ----------
    f : function                                                                                                                                         
        Input function for which the zero is to be found.
    a : real number
        Left side of interval.
    b : real number
        Right side of interval.
    Nmax : integer
        Number of iterations to be performed.
        
    Returns
    -------
    p_array : numpy.ndarray, shape (Nmax,)
        Array containing the sequence of approximations.
    """
    
    # Initialise the array with zeros
    p_array = np.zeros(Nmax)
    
    # Continue here:...
    if f(a) * f(b) < 0:
        for i in range(Nmax):
            p = (a + b) / 2
            if np.sign(f(p)) == np.sign(f(a)):
                a = p
            else:
                b = p
            p_array[i] = p
            
    else:
        raise Exception ('The scalars a and b do not bound a root.')
     
    return p_array


def fixedpoint_iteration (f ,c , p0 , Nmax ):
    
    """
    Fixed Point Iteration Method: Returns a numpy array of the 
    sequence of approximations obtained by the fixed point iteration method.
    
    Parameters
    ----------
    f : function
        Input function for which the zero is to be found.
    c : real number
        Coefficient of f(x), defined in g(x) = x - c f(x).
    p0 : real number
        Initial approximation to start the fixed-point iteration.
    Nmax : integer
        Number of iterations to be performed.
        
    Returns
    -------
    p_array : numpy.ndarray, shape (Nmax,)
        Array containing the sequence of approximations.
    """
    # Initialise the array with zeros
    p_array = np.zeros(Nmax)
    
    for i in range(Nmax):
        p = p0 - c * f(p0)
        p_array[i] = p
        p0 = p
    
    return p_array


def newton_method (f , dfdx , p0 , Nmax ):
    p_array = np.zeros(Nmax)
    
    for i in range(Nmax):
        p = p0 - f(p0)/dfdx(p0)
        p0 = p
        p_array[i] = p0
    return p_array


def plot_convergence ( p_exact ,f , dfdx ,c , p0 , p1 , Nmax , fig ):
    p_array1 = bisection(f,p0,p1,Nmax)
    p_array2 = fixedpoint_iteration(f ,c , p0 , Nmax )
    p_array3 = newton_method (f , dfdx , p0 , Nmax )
    p_array4 = secant_method (f , p0 , p1 , Nmax )
    
    plt.semilogy(np.abs(p_exact - p_array1),"o")
    plt.semilogy(np.abs(p_exact - p_array2),"d")
    plt.semilogy(np.abs(p_exact - p_array3),"s")
    plt.semilogy(np.abs(p_exact - p_array4),"v")
    plt.legend(["bisection","fixed-point iteration","Newton's method","Secant method"], loc='upper right', prop={'size': 6})
    return None

def secant_method (f , p0 , p1 , Nmax ):
    p_array = np.zeros(Nmax)
    
    for i in range(Nmax):
        
        if np.abs(f(p1) - f(p0)) < 1e-14:
            p = p1
            p_array[i] = p
            p0 = p1
        else:
            p = p1 - f(p1) / ((f(p1)-f(p0)) / (p1-p0))
            p_array[i] = p
            p0 = p1
            p1 = p
    return p_array
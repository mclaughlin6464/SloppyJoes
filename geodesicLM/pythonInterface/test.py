from geodesiclm import geodesiclm
from time import time
import numpy as np


def rosenbrock(x, A = 10):
    """ Rosenbrock function """
    """ solution: all ones """
 
    n = x.shape[0]
    outs = []
    for i in xrange(n-1):
        outs.append(1-x[i])
        outs.append(A * (x[i+1] - x[i]**2))

    return np.array( outs )

def polynomial_fit(theta, x, y):
    y_hat = np.sum([t*(x**p) for p, t in enumerate(theta)],axis=0) 
    return y_hat - y 

def exponential_fit(theta, x, y, coeff=None):
    if coeff is None:
        coeff = np.ones(theta.shape)
    y_hat = np.sum([coeff[i]*np.exp(-x*theta[i]) for i in xrange(theta.shape[0])], axis=0) 
    return y_hat - y

def model_fit(theta,x,y, model_type='poly'):
    if(model_type=='poly'):
        return polynomial_fit(theta, x,y)
    if(model_type=='exp'):
        if(type(theta)==tuple and len(theta) > 1):
            return exponential_fit(theta[0], x,y,theta[1])
        else:
            return exponential_fit(theta, x,y)

def lbfgs_test(theta, x, y):
    y_hat = (theta[0]*x)**2 + theta[1]*theta[0]*x+theta[1]**2
    return y_hat-y

def paraboloid(x):
    """ Parabaloid  """
    """ Solution all 0's."""
    return np.array(x)

def beale(x):
    """ Beale function """
    """ solution: [3, 0.5] """ 
    
    # nonconvex - x[0] should initially be positive
    if (np.max(abs(x)) > 4.5):
        return np.array([10,10,10]) # something bad
 
    v1 = 1.5 - x[0] + x[0] * x[1]
    v2 = 2.25 - x[0] + x[0] * x[1]**2
    v3 = 2.625 - x[0] + x[0] * x[1]**3

    return np.array( [v1, v2, v3] ) 


#x0 = np.array(np.random.rand(2))
np.random.seed(int(time()))
x0 = np.array([-1.0, -1.0, 1.0, -1.0])
print rosenbrock(x0)
xf, info = geodesiclm(rosenbrock, x0, args = (10,), full_output=1, print_level = 5, iaccel = 1, maxiters = 10000, artol = -1.0, xtol = -1, ftol = -1, avmax = 2.0, factoraccept=3.0, factorreject=5.0)
print info

print xf

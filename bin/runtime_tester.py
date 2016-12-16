from __future__ import print_function
from SloppyJoes import lazy_wrapper
import numpy as np
import time
from test import rosenbrock, paraboloid
import sys
import matplotlib.pyplot as plt

minparams = 2
maxparams = 7
ntrials = 2
results = {}
func = rosenbrock
maxiter = 10 

for nparam in range(minparams,maxparams):
  approx_start = time.time() 
  res = [] 
  for trial in range(ntrials):
    x0 = 1 - 2*np.random.rand(nparam)
    xf = lazy_wrapper(func, x0, args = (), full_output=1, print_level = 0, iaccel = 1, maxiters = maxiter, approx = 'LBFGS')
    res.append(np.sum(xf**2))
    
  approx_time = time.time() - approx_start
  results[nparam] = {} 
  results[nparam]['approx_time'] = approx_time
  results[nparam]['yval_approx'] = np.array(res) 
 
  exact_start = time.time() 
  res = [] 
  for trial in range(ntrials):
    x0 = 1 - 2*np.random.rand(nparam)
    xf = lazy_wrapper(func, x0, args = (), full_output=1, print_level = 0, iaccel = 1, maxiters = maxiter)
    res.append(np.sum(xf**2))
    
  exact_time = time.time() - exact_start 
  results[nparam]['exact_time'] = exact_time
  results[nparam]['yval'] = np.array(res) 
   
  print('Finished with ' + str(nparam) + ' parameters',file=sys.stderr)
  sys.stderr.flush()

timevec = np.array([results[i]['approx_time']/ntrials for i in range(minparams,maxparams)])
extimevec = np.array([results[i]['exact_time']/ntrials for i in range(minparams,maxparams)])
plt.plot(range(minparams,maxparams),timevec)
resvec = [np.mean(results[i]['yval']) for i in range(2,maxparams)]
#plt.plot(range(2,maxparams),extimevec*np.max(timevec)/np.max(extimevec),c='r')
plt.plot(range(minparams,maxparams),extimevec,c='r')
plt.xlabel('Number of parameters')
plt.ylabel('Mean optimization time (s)')
plt.title('Optimization of the Rosenbrock function')
plt.legend(['Approximate method', 'Geodesic Acceleration'],loc=2)
#plt.savefig('compare1.png',dpi=256)
plt.show()

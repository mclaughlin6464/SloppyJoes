
from geodesiclm import geodesiclm
import numpy as np
import time
from test import rosenbrock
import sys

ntrials = 10
results = {}
for nparam in range(2,80):
  total_start = time.time() 
  res = [] 
  iters = [] 
  evals = [] 
  for trial in range(ntrials):
    x0 = 2 - 4*np.random.rand(nparam)
    xf, info = geodesiclm(rosenbrock, x0, args = (), full_output=1, print_level = 0, iaccel = 1, maxiters = 10000, artol = -1.0, xtol = -1, ftol = -1, avmax = 2.0)
    res.append(xf)
    iters.append(info['iters'][0])
    evals.append(info['iters'][1]) 
   
  total_time = time.time() - total_start
  results[nparam] = {} 
  results[nparam]['avg_iters'] = iters
  results[nparam]['evals'] = evals
  results[nparam]['time'] = total_time
  results[nparam]['xval'] = np.array(res) 
  
  print 'Finished with ' + str(nparam) + ' parameters'
  sys.stdout.flush()

plt.plot(range(2,80),[results[i]['time']/ntrials for i in range(2,80)])
plt.show()

#!/bin/bash
'''Python rewrite of the accept module'''
__author__ = 'Sean McLaughlin'
__email__ = 'swmclau2@stanford.edu'

import numpy as np

def acceptance(C, Cnew, Cbest, ibold, accepted, dtd, v, vold):

    if Cnew <= C: #accept all downhill steps
        accepted = max(accepted+1, 1)
    else:
        #calculate beta
        if np.sum(vold**2) == 0:
            beta = 1.0
        else:
            beta = np.dot(v, np.dot(dtd, vold))
            beta = beta/np.sqrt(np.dot(v, np.dot(dtd,v))*np.dot(vold, np.dot(dtd, vold)))
            beta = min(1.0, 1.0-beta)

        if ibold == 0: #only donwhill steps
            #why would this be accessed?
            if Cnew <= C:
                accepted = max(accepted+1,1)
            else:
                accepted = min(accepted-1, -1)

        elif ibold == 1:
            if beta*Cnew <= Cbest:
                accepted = max(accepted + 1, 1)
            else:
                accepted = min(accepted - 1, -1)

        elif ibold == 2:
            if beta*beta * Cnew <= Cbest:
                accepted = max(accepted + 1, 1)
            else:
                accepted = min(accepted - 1, -1)

        elif ibold == 3:
            if beta * Cnew <= C:
                accepted = max(accepted + 1, 1)
            else:
                accepted = min(accepted - 1, -1)

        elif ibold == 4:
            if beta * beta * Cnew <= C:
                accepted = max(accepted + 1, 1)
            else:
                accepted = min(accepted - 1, -1)

        if accepted < 0:
            print 'Rejected acceptance'
            print Cnew, C, Cbest

    return accepted
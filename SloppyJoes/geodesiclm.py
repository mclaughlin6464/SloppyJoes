#!/bin/bash
''' Python rewrite of the geodesicLM main module. '''
__author__ = 'Sean McLaughlin'
__email__ = 'swmclau2@stanford.edu'

import numpy as np
from scipy import linalg

from .fdjac import fdjac
from .fdavv import fdavv
from .lambdaFuncs import TrustRegion, Updatedelta_factor, Updatedelta_more, Updatelam_factor, Updatelam_nelson, \
    Updatelam_Umrigar
from .updatejac import updatejac
from .lbfgs import update_storage
from .accept import acceptance
from .converge import convergence_check

#Most of these can be removed, and also absorbed into a kwarg
#i'll make notes of which ones that is
def geodesiclm(func, jacobian, Avv, x, fvec, fjac, n,m,k,callback,info,\
               analytic_jac, analytic_Avv, center_diff, h1, h2, \
                dtd, damp_mode, niters, nfev, njev, jaev, maxiter, maxfev, maxjev,
               maxaev, maxlam, minlam, artol, Cgoal, gtol, xtol, xrtol, ftol, frtol,
               converged, print_level, print_unit, imethod, iaccel, ibold, ibroyden, initialfactor,\
               factoraccept, factorreject, avmax):

    converged_info = {}

    converged_info[1] = 'artol reached'
    converged_info[2] = 'Cgoal reached'
    converged_info[3] = 'gtol reached'
    converged_info[4] = 'xtol reached'
    converged_info[5] = 'xrtol reached'
    converged_info[6] = 'ftol reached'
    converged_info[7] = 'frtol reached'
    converged_info[-1] = 'maxiters exeeded'
    converged_info[-2] = 'maxfev exceeded'
    converged_info[-3] = 'maxjev exceeded'
    converged_info[-4] = 'maxaev exceeded'
    converged_info[-10] = 'User Termination '
    converged_info[-11] = 'NaN Produced'

    if print_level >=1:
        output_string = '''
        Optimizing with Geodesic-Levenberg-Marquardt algorithm, version 1.1\n
        Method Details:
        Update method: %d
        acceleration: %d
        Bold method: %d
        Broyden updates: %d'''%(imethod, iaccel, ibold, ibroyden)
        print output_string

    niters = 0
    nfev, naev, njaev = 0,0,0
    #think this can be boolean
    converged = 0

    v = np.zeros(n)
    vold = np.zeros_like(v)
    a = np.zeros_like(v)

    cos_alpha = 1.0
    av = 0
    a_param = 0.5

    #initialize our variable storage
    s= np.zeros((n,k))
    y = np.zeros_like(s)
    #not sure if these really need to be initialized
    neg_delta_C, neg_delta_C_old = np.zeros(n), np.zeros(n)
    alpha = 0.005
    n_accepted = 0

    H_0 = np.eye(n)*0.001
    dtd_inv = np.eye(n)

    accepted=0
    counter = 0

    #CALL func(m,n,x,fvec)
    fvec = func(x)

    nfev+=1
    C = 0.5*np.sum(fvec**2) #probably needs to have an axis, depening on shape of fvec

    if print_level >=1:
        print 'Initial Cost: %0.3f'%C

    if np.any(np.isnan(fvec)):
        converged = -11
        maxiter = 0

    Cbest = C
    fvec_best = fvec
    x_best = x

    if analytic_jac:
        fjac = jacobian(x)
        njev+=1
    else:
        fjac = fdjac(x, fvec, func, h1, center_diff)
        nfev = nfev + 2*n if center_diff else nfev + n

    jac_uptodate = True
    jac_force_update = False
    jtj = np.dot(fjac.T, fjac)

    acc = np.zeros(m) #could move up with the other initializations

    if damp_mode==0:
        dtd = np.eye(n)
    elif damp_mode==1:
        dtd[np.diag_indices(dtd.shape[0])] = np.max(np.vstack([np.diag(jtj), np.diag(dtd)]), axis=0)

    if imethod < 10:
        lam = np.max(np.diag(jtj))*initialfactor
    elif imethod >=10:
        delta = initialfactor*np.sqrt(np.dot(x, np.dot(dtd,x)))
        delta = 100.0 if delta == 0.0 else delta
        lam = 1.0

        if converged == 0:
            #Not sure if this only returns lambda
            lam = TrustRegion(fvec, fjac, dtd, delta)

    fvec_new = np.zeros_like(fvec)#i think have to initialize this here
    x_new = np.zeros_like(x) #not sure about these initializations
    Cnew, Cold = np.inf, np.inf

    #main loop
    for istep in xrange(maxiter):

        info = callback(x,v,a,fvec, fjac, acc, lam, dtd, fvec_new, accepted)
        if info != 0:
            converged = -10
            break

        #full or parital jac update
        if accepted>=0 and ibroyden <=0:
            jac_force_update = True
        if accepted+ibroyden <= 0 and not jac_uptodate:
            jac_force_update = True

        if accepted > 0 and ibroyden > 0 and not jac_force_update: #rank deficient
            fjac = updatejac(fjac, fvec, fvec_new, acc, v, a)
            jac_uptodate = False

        if accepted > 0: #accepted step

            #update lbfgs
            fvec = fvec_new

            # TODO this will always be true!
            successful_update = update_storage(k,n_accepted+1, s, y, x_new-x, neg_delta_C_old-neg_delta_C)
            if successful_update: #accepted
                n_accepted+=1

            neg_delta_C_old = neg_delta_C
            x = x_new
            vold = v
            C = Cnew

            if C <= Cbest:
                x_best = x
                Cbest = C
                fvec_best = fvec

        #TODO could have this as a helper function as its repeated from above
        if jac_force_update: #full rank update of the jacobian
            if analytic_jac:
                fjac = jacobian(x)
                njev += 1
            else:
                fjac = fdjac(x, fvec, func, h1, center_diff)
                nfev = nfev + 2 * n if center_diff else nfev + n

        valid_result = np.all(~np.isnan(fjac))

        if valid_result: #no nans, lets party

            jtj = np.dot(fjac.T, fjac) #not sure if this is the write shape

            #update scaling/lam/trustregion
            if istep > 1:
                if damp_mode == 1:
                    dtd[np.diag_indices(dtd.shape[0])] = np.max(np.vstack([np.diag(jtj), np.diag(dtd)]), axis=0)
                #could write a helper function to wrap this up
                if imethod == 0:
                    #update lam directly by fixed factors
                    lam = Updatelam_factor(lam, accepted, factoraccept, factorreject)
                elif imethod == 1:
                    #TODO
                    #update based on gain factor rho (see Nielson reference)
                    lam = Updatelam_nelson(lam, accepted, factoraccept, factorreject, rho)
                elif imethod == 2:
                    #update lam directly using method of Umrigar and Nightingale
                    lam = Updatelam_Umrigar(lam, accepted, v, vold, fvec, fjac, dtd, a_param, Cold, Cnew)
                elif imethod == 10:
                    #update delta by fixed factors
                    delta = Updatedelta_factor(delta, accepted, factoraccept, factorreject)
                    lam = TrustRegion(fvec, fjac, dtd, delta)
                elif imethod == 11:
                    delta, lam = Updatedelta_more(delta, lam, x, dtd, rho, C, Cnew, dirder, actred, av, avmax)
                    lam = TrustRegion(fvec, fjac, dtd, delta)


            #propose step

            g = jtj+lam*dtd
            H_0[np.diag_indices_from(H_0)] = 1.0/np.diag(g)

            dtd_inv[np.diag_indices_from(dtd)] = 1.0/(lam*np.diag(dtd))

            g_upper = linalg.cholesky(g)
            info = 0
        else:#nans in jac
            converged = -11
            break

        if info==0: #decomp successful
            neg_delta_C = -np.dot(fvec, fjac)

            #for now, just get the original version without my additions
            v = linalg.cho_solve((g_upper, False), neg_delta_C)

            temp1 = 0.5*np.dot(v, np.dot(jtj,v))/C
            temp2 = 0.5*lam*np.dot(v, np.dot(dtd,v))/C
            pred_red = temp1+2*temp2
            dirder = -1*(temp1+temp2)
            #calculate cos_alpha, - cos of anlge between step direction and residual
            cos_alpha = np.abs(np.dot(fvec, np.dot(fjac, v)))
            cos_alpha = cos_alpha/np.sqrt(np.dot(fvec, fvec)*np.dot(np.dot(fjac,v), np.dot(fjac, v)))

            if imethod < 10:
                delta = np.sqrt(np.dot(v, np.dot(dtd, v))) #update delta if not set directly

            #update acceleration!

            if iaccel > 0:
                if analytic_Avv:
                    acc = Avv(x,v)
                    naev+=1
                else:
                    acc = fdavv(x,v,fvec, fjac, func, jac_uptodate, h2)
                    nfev = nfev+1 if jac_uptodate else nfev + 2

            valid_result = np.all(~np.isnan(acc))
            #acceleration calc
            if valid_result:
                a = -np.dot(acc, fjac) #dont have a good name for this
                a = linalg.cho_solve((g_upper, False), a)

            else:
                a = np.zeros_like(v)

            #evaluate at proposed step if av <=avmax
            av = np.sqrt(np.dot(a, np.dot(dtd, a))/np.dot(v, np.dot(dtd, v)))
            if av <= avmax:
                x_new = x+v+0.5*a
                fvec_new = func(x_new)
                nfev+=1
                Cnew = 0.5*np.sum(fvec**2)
                Cold = C

                valid_result = np.all(~np.isnan(fvec_new))

                if valid_result: #proceed as normal
                    actred = 1.0-Cnew/C
                    rho = 0.0
                    if pred_red != 0:
                        rho = actred/pred_red
                    accepted = acceptance(C, Cnew, Cbest, ibold, accepted, dtd, v, vold)
                else: #reject step
                    actred = 0.0
                    rho = 0.0
                    accepted = min(accepted-1, -1)

            else: #acceleration too large, reject
                accepted = min(accepted-1, -1)
        else:
            accepted = min(accepted-1, -1)

        if converged == 0:
            counter, converged = convergence_check(converged, accepted, counter, C, Cnew, x, fvec, fjac, lam, x_new,
                      nfev, maxfev, njev, maxjev, naev, maxaev, maxlam, minlam, artol,
                      Cgoal, gtol, xtol, xrtol, ftol,frtol, cos_alpha)
            if converged == 1 and not jac_uptodate:
                #if converged by artol with an out of date jacobian update the jacobian to confirm true conergence
                converged = 0
                jac_force_update = True

        #printing
        if print_level == 2 and accepted > 0:
            print "  istep, nfev, njev, naev, accepted", istep, nfev, njev, naev, accepted
            print "  Cost, lam, delta", C, lam, delta
            print "  av, cos alpha", av, cos_alpha
        elif print_level == 3:
            print "  istep, nfev, njev, naev, accepted", istep, nfev, njev, naev, accepted
            print "  Cost, lam, delta", C, lam, delta
            print "  av, cos alpha", av, cos_alpha
        elif print_level == 4 and accepted > 0:
            print "  istep, nfev, njev, naev, accepted", istep, nfev, njev, naev, accepted
            print "  Cost, lam, delta", C, lam, delta
            print "  av, cos alpha", av, cos_alpha
            print "  x = ", x
            print "  v = ", v
            print "  a = ", a
        elif print_level == 5:
            print "  istep, nfev, njev, naev, accepted", istep, nfev, njev, naev, accepted
            print "  Cost, lam, delta", C, lam, delta
            print "  av, cos alpha", av, cos_alpha
            print "  x = ", x
            print "  v = ", v
            print  "  a = ", a

        if converged != 0:
            break#w'ere done!

        if accepted >0:
            jac_uptodate = False #jaobian out of date

    if converged == 0: #not converged
        conveged = -1

    niters = istep
    #return best fit found

    x = x_best
    fvec = fvec_best

    if print_level >= 1:
        print "Optimization finished"
        print "Results:"
        print "  Converged:    ", converged_info[converged], converged
        print "  Final Cost: ", 0.5 * np.dot(fvec, fvec)
        print "  Cost/DOF: ", 0.5* np.dot(fvec, fvec) / (m - n)
        print "  niters:     ", istep
        print "  nfev:       ", nfev
        print "  njev:       ", njev
        print "  naev:       ", naev

    return x_best

def lazy_wrapper(func, x0, **kwargs):
    """
    Arguments for leastsq:
    (note the only required arguments are func and x0, all others will be set to a default value)

    func -- a routine for calculting the residuals
    func_args -- a tuple of arguments to pass to func

    x0 -- the initial parameter guess (passed as a numpy array)

    jacobian -- a routine for calculating the jacobian matrix.  If not passed, a finite difference estimate is used.
    jacobian_args -- a tuple of arguments to pass to jacobian
    h1 -- controls the step size for calculating a finite difference estimate of the jacobian

    Avv -- a routine for calculating the directional second derivative
    Avv_args -- a tuple of arguments to pass to Avv.  Avv will be called as Avv(x,v,*Avv_args) where x is the parameter guess and v is the direction to calculate the second derivative
    h2 -- controls the step size for calculating a finite difference estimate of the directional second derivative

    eps -- a number estimating the accuracy to which the function is evaluated

    maxiters -- a list of integers [maxiter, maxfev, maxjev, maxaev] (the maximum number of allowed iterations, function evaluations, jacobian evaluations, and acceleration evaluations)

    tols -- a list of floats controlling stopping criteria: [artol, Cmin, gtol, xtol, xrtol, ftol, frtol, maxlam]
       artol = cosine of the angle between the unfit residuals and the range of the jacobian -- typically set to ~.01
       Cmin = stop when the cost is less than this value
       gtol = stop when the gradient is less than this value
       xtol = stop when the step size becomes smaller than this value
       xrtol = stop when the relative change in each parameter is less than this value
       ftol = stop when the cost decreases by less than this value for 3 consecutive iterations
       frtol = stop when the relative decrease in the cost is less than this value for 3 consecutive iterations
       maxlam = stop when the damping term is larger than maxlam
       minlam = stop when the damping term is smaller than minlam for 3 consective steps

    print_level -- an integer indicating how much information to print, ranges from 0 to 5 (higher number prints mor details).  Typically only needed for debugging

    method_flags -- a list of integers controlling details of the algorithm (imethod, iaccel, ibold, ibroyden).  See documentation in geodesiclm code for details about these parameters

    method_params -- a list of floats controlling details of the algorithm (factor_initial, factor_accept, factor_reject, avmax).  See documentation in geodesiclm code for details.

    callback -- a callback function to be called after each iteration

    m -- an integer specifying the number of residuals
    """

    ## Check for kwargs

    ## func_args
    if kwargs.has_key('func_args'):
        func_extra_args = kwargs['func_args']
    elif kwargs.has_key('args'):
        func_extra_args = kwargs['args']
    else:
        func_extra_args = ()

    ## jacobian, h1, jacobian_extra_args
    if kwargs.has_key('jacobian'):
        jacobian = kwargs['jacobian']
        h1 = -1.0
        if kwargs.has_key('jacobian_args'):
            jacobian_extra_args = kwargs['jacobian_args']
        elif kwargs.has_key('args'):
            jacobian_extra_args = kwargs['args']
        else:
            jacobian_extra_args = ()
        analytic_jac = True
    else:
        jacobian = jacobian_dummy
        jacobian_extra_args = ()
        if kwargs.has_key('h1'):
            h1 = kwargs['h1']
        else:
            h1 = 1.49012e-08
        analytic_jac = False

    ## Avv, h2, Avv_args
    if kwargs.has_key('Avv'):
        Avv = kwargs['Avv']
        h2 = -1.0
        if kwargs.has_key('Avv_args'):
            Avv_extra_args = kwargs['Avv_args']
        elif kwargs.has_key('args'):
            Avv_extra_args = kwargs['args']
        else:
            Avv_extra_args = ()
        analytic_Avv = True
    else:
        Avv = Avv_dummy
        Avv_extra_args = ()
        if kwargs.has_key('h2'):
            h2 = kwargs['h2']
        else:
            h2 = 0.1
        analytic_Avv = False

    ## center_diff
    if kwargs.has_key('center_diff'):
        center_diff = kwargs['center_diff']
    else:
        center_diff = False

    ## callback
    if kwargs.has_key('callback'):
        callback = kwargs['callback']
    else:
        callback = callback_dummy

    ## info
    info = 0

    ## dtd
    dtd = np.empty( (len(x0), len(x0) ), order = 'F')
    if kwargs.has_key('dtd'):
        dtd[:,:] = kwargs['dtd'][:,:] # guarantee that order = 'F'
    else:
        dtd[:,:] = np.eye( len(x0) )[:,:]

    if kwargs.has_key('damp_mode'):
        damp_mode = kwargs['damp_mode']
    else:
        damp_mode = 1

    ## maxiter
    if kwargs.has_key('maxiter'):
        maxiter = kwargs['maxiter']
    else:
        maxiter = 200*(len(x0) + 1 )

    ## maxfev
    if kwargs.has_key('maxfev'):
        maxfev = kwargs['maxfev']
    else:
        maxfev = 0

    ## maxjev
    if kwargs.has_key('maxjev'):
        maxjev = kwargs['maxjev']
    else:
        maxjev = 0

    ## maxaev
    if kwargs.has_key('maxaev'):
        maxaev = kwargs['maxaev']
    else:
        maxaev = 0

    ## maxlam
    if kwargs.has_key('maxlam'):
        maxlam = kwargs['maxlam']
    else:
        maxlam = -1.0

    ## minlam
    if kwargs.has_key('minlam'):
        minlam = kwargs['minlam']
    else:
        minlam = -1.0

    ## artol
    if kwargs.has_key('artol'):
        artol = kwargs['artol']
    else:
        artol = 0.001

    ## Cgoal
    if kwargs.has_key('Cgoal'):
        Cgoal = kwargs['Cgoal']
    else:
        #Cgoal = 1.49012e-08
	Cgoal = 1.0e-05

    ## gtol
    if kwargs.has_key('gtol'):
        gtol = kwargs['gtol']
    else:
        gtol = 1.49012e-08

    ## xtol
    if kwargs.has_key('xtol'):
        xtol = kwargs['xtol']
    else:
        xtol = 1.49012e-08

    ## xrtol
    if kwargs.has_key('xrtol'):
        xrtol = kwargs['xrtol']
    else:
        xrtol = -1.0

    ## ftol
    if kwargs.has_key('ftol'):
        ftol = kwargs['ftol']
    else:
        ftol = 1.49012e-08

    ## frtol
    if kwargs.has_key('frtol'):
        frtol = kwargs['frtol']
    else:
        frtol = -1.0

    ## print_level
    if kwargs.has_key('print_level'):
        print_level = kwargs['print_level']
    else:
        print_level = 0

    ## print_unit
    print_unit = 6

    if kwargs.has_key('imethod'):
        imethod = kwargs['imethod']
    else:
        imethod = 0

    ## iaccel
    if kwargs.has_key('iaccel'):
        iaccel = kwargs['iaccel']
    else:
        iaccel = 1

    ## ibold
    if kwargs.has_key('ibold'):
        ibold = kwargs['ibold']
    else:
        ibold = 2

    ## ibroyden
    if kwargs.has_key('ibroyden'):
        ibroyden = kwargs['ibroyden']
    else:
        ibroyden = 0

    ## initialfactor
    if kwargs.has_key('initialfactor'):
        initialfactor = kwargs['initialfactor']
    else:
        if imethod < 10:
            initialfactor = 0.001
        else:
            initialfactor = 100.0

    ## factoraccept
    if kwargs.has_key('factoraccept'):
        factoraccept = kwargs['factoraccept']
    else:
        factoraccept = 3.0

    ## factorreject
    if kwargs.has_key('factorreject'):
        factorreject = kwargs['factorreject']
    else:
        factorreject = 2.0

    ## avmax
    if kwargs.has_key('avmax'):
        avmax = kwargs['avmax']
    else:
        avmax = 0.75

    ## m
    if kwargs.has_key('m'):
        _m = kwargs['m']
    else:
        _m = len(func(x0,*func_extra_args))
    print _m

    ## LBFGS-memory
    if kwargs.has_key('k'):
        k = kwargs['k']
    else:
        k = 10

    #not sure about these
    fvec = np.empty((_m,))
    fjac = np.empty((_m,len(x0)),order = 'F')

    niters = np.empty((1,), dtype = np.int32)
    nfev = np.empty((1,), dtype = np.int32)
    njev = np.empty((1,), dtype = np.int32)
    naev = np.empty((1,), dtype = np.int32)
    converged = np.empty((1,),dtype=np.int32)

    x = x0.copy()

    f = lambda x: func(x, *func_extra_args)

    return geodesiclm(f, jacobian, Avv, x, fvec, fjac, x.shape[0],_m,k,callback,info,\
               analytic_jac, analytic_Avv, center_diff, h1, h2, \
                dtd, damp_mode, niters, nfev, njev, naev, maxiter, maxfev, maxjev,
               maxaev, maxlam, minlam, artol, Cgoal, gtol, xtol, xrtol, ftol, frtol,
               converged, print_level, print_unit, imethod, iaccel, ibold, ibroyden, initialfactor,\
               factoraccept, factorreject, avmax)
                           #func_extra_args = func_extra_args,
                           #jacobian_extra_args = jacobian_extra_args,
                           #Avv_extra_args = Avv_extra_args)

def jacobian_dummy(x,*args):
    pass

def Avv_dummy(x,v,*args):
    pass

def callback_dummy(x,v,a,fvec,fjac,acc,lam,dtd,fvec_new,accepted):
    return 0

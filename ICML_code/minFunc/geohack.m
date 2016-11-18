function results = geohack(funObj, x0, options, varargin)

maxIter = getOpt(options,'MAXITER',500);

% this will need some work
% first (two?) inputs to python() specify python script location
% then specify arguments
[opt_x, status] = python('../../geodesicLM/pythonInterface/geo_wrapper', 'geo_wrapper.py', funObj, x0, ...
        args = varargin, full_output=0, print_level = 5, iaccel = 1, maxiters = maxIter, artol = -1.0, xtol = -1, ftol = -1, avmax = 2.0);

results = {opt_x, funObj(opt_x)};
end

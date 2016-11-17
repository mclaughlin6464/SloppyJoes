global usegpu gpusingletype singletype;
global selgpu;
addpath (genpath('/afs/cs/u/jngiam/MATLAB/GPUmat/'))
usegpu = 'gpumat';
selgpu = GPUautostart;
gpusingletype = @GPUsingle;
singletype    = @GPUsingle;
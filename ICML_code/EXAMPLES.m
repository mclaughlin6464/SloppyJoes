%This file contains an example illustrating how you should use this
%package and call the localrfAutoencoderWrapper function.

optimization_option = 1; %L-BFGS
batchsize = 10000; % size of minibatch used during optimization. We got best results with this value.
maxinner = 50; %The maximum number of iterations you want to run the optimization for a fixed minibatch, 
               %before swapping the current minibatch out and using another
               %one. We found a value of around 50 to work the best for
               %L-BFGS
gpu = 1; %1 to use gpu, 0 to use cpu.
machine = 'gorgon1';
numhidden = 10000; %number of hidden neurons in the autoencoder model.
lambda = 0.0001; %The weight of the weight regularization term in the objective function described in the paper.
gamma = 10; %The weight of the sparsity term in the objective function described in the paper.
alpha = 0;
beta = 0;
momentum = 0;
targetact = 0.001; %The target activation value for each of the neurons in the hidden layer.
windowSize = 28; %The size of the receptive field for each hidden neuron.
step = 1; %the stride size between 2 consecutive hidden neuron's receptive fields(how many pixels we can skip), default is 1
outside = 0; %keeping this 0 will force all receptive field to have the same side
rpcoptions = struct(); %We are not doing parallel runs, so no need to define an rcpoptions struct
corrections = 10; %we found this value of corrections to work well for L-BFGS
tSgd = 0;
resume = 0;
gpuDriver = 1; %its best if you keep this to 1 if you are using gpu(so that you use jacket)
savePath = './results/'; % please create a directory called results 
dataPath = './data/'; % please put your mnist dataset in here
jacketPath = '/usr/jacket/engine'; %please replace this with the directory where the jacket engine is stored.
localrfAutoencoderWrapper(optimization_option, batchsize, maxinner, gpu, machine, numhidden, lambda, gamma, alpha, beta, momentum, targetact, windowSize, step, outside, rpcoptions, corrections, tSgd, resume, gpuDriver, savePath, dataPath, jacketPath)

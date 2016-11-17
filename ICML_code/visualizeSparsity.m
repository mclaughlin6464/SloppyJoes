clear all
load results/autoencoder_o1_s40000_i20_lbfgs_gpu0_larva_n100.mat
theta = params;
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
plotrf(W1', 10, [])

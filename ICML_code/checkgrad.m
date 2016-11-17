clear all; close all;
visibleSize = 10;
hiddenSize = 11;
data = randn(visibleSize, 100);
theta = initializeParameters(hiddenSize, visibleSize);
[cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, data, 10, 0.0001, 0, 0.5, 1, 0.001);
eps = 1e-5;
for i=1:length(theta)
    newtheta = theta;
    newtheta(i) = newtheta(i) + eps;
    cost1 = sparseAutoencoderCost(newtheta, visibleSize, hiddenSize, data, 10, 0.0001, 0, 0.5, 1, 0.001);
    newtheta = theta;
    newtheta(i) = newtheta(i) - eps;
    cost2 = sparseAutoencoderCost(newtheta, visibleSize, hiddenSize, data, 10, 0.0001, 0, 0.5, 1, 0.001);
    numgrad = (cost1 - cost2)/(2*eps);
    fprintf('grad(i) = %f, numgrad(i) = %f, diff = %f\n', grad(i), numgrad, abs(numgrad - grad(i)));

end
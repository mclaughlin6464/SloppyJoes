function [cost,grad] = parallelSparseAutoencoderCost(theta, server, visibleSize, hiddenSize, beta, lambda, gpu, batchindex, rhoStar, numcases, targetact)
if nargout > 1
    [cost, grad] = server.rpc('clientBigSparseAutoencoderCost', theta, visibleSize, hiddenSize, beta, lambda, gpu, batchindex, rhoStar, numcases, targetact);
    cost = sum(cell2mat(reshape(cost,1,1,server.slaveCount)),3)/server.slaveCount;
    grad = sum(cell2mat(reshape(grad,1,1,server.slaveCount)),3)/server.slaveCount;                           
else
    cost = server.rpc('clientBigSparseAutoencoderCost', theta, visibleSize, hiddenSize, beta, lambda, gpu, batchindex, rhoStar, numcases, targetact);
    cost = sum(cell2mat(reshape(cost,1,1,server.slaveCount)),3)/server.slaveCount;                         
end
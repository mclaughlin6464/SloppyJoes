function [newparams] = collapseParams(params, visibleSize, hiddenSize, indices, indexVector)
W1 = reshape(params(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W1 = collapseWeightsFull(W1, indices, indexVector);
newparams = [W1(:); params(hiddenSize*visibleSize+1:end)];
end

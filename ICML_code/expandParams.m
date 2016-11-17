function [newparams] = expandParams(params, compressedHiddenSize, indices, indexVector)
W1 = params(1:compressedHiddenSize);
W1 = reshape(W1, size(indices,1), compressedHiddenSize/size(indices,1));
W1 = expandWeightsFull(W1, indices, indexVector);
newparams = [W1(:); params(compressedHiddenSize+1:end)];
end

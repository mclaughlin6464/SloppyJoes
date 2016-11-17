function [cost,grad] = clientBigSparseAutoencoderCost(oldtheta, visibleSize, hiddenSize, beta, lambda, gpu, batchindex, rhoStar, numcases, targetact)
global state;
theta = expandParams(oldtheta, state.compressedHiddenSize, state.indices, state.indexVector);
data = state.data(:, :, mod(batchindex, state.numbatches)+1);
weighting = size(data,2)/numcases;
if size(data, 2) <= 10000
    if nargout > 1
        [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             data, beta, lambda, gpu, rhoStar, weighting, targetact);
    else
        cost = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             data, beta, lambda, gpu, rhoStar, weighting, targetact);
    end
else
    batchsize = 10000;
    l = size(data,2);
    numbatches = ceil(l/ batchsize);
    if nargout > 1
        cost = 0;
        grad = 0;
        for i=1:numbatches
            batchdata = data(:, (i-1)*batchsize + 1:min(i*batchsize, l));
            size(batchdata)
            [batchcost,batchgrad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                batchdata, beta, lambda, gpu, rhoStar, weighting, targetact);
            cost = cost + batchcost;
            grad = grad + batchgrad;
        end
    else
        cost = 0;
        for i=1:numbatches
            batchdata = data(:, (i-1)*batchsize + 1:min(i*batchsize, l));
            batchcost = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                batchdata, beta, lambda, gpu, rhoStar, weighting, targetact);
            cost = cost + batchcost;
        end
    end
end
if nargout > 1
    [grad] = collapseParams(grad, visibleSize, hiddenSize, state.indices, state.indexVector);
end
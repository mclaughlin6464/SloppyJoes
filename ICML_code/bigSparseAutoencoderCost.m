function [cost,grad] = bigSparseAutoencoderCost(oldtheta, visibleSize, hiddenSize, ...
                                             data, beta, lambda, gpu, rhoStar, weighting, targetact, compressedHiddenSize, indices, indexVector)
%This function splits up the data into smaller batches to fit gpu memory
%and computes the objective value for the current estimate of theta.
theta = expandParams(oldtheta, compressedHiddenSize, indices, indexVector);
if gpu == 0
    if nargout > 1
        [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             data, beta, lambda, gpu, rhoStar, weighting, targetact);
    else
        cost = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             data, beta, lambda, gpu, rhoStar, weighting, targetact);
    end
else
    if size(data, 2) <= 5000
        if nargout > 1
            [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             data, beta, lambda, gpu, rhoStar, weighting, targetact);
        else
            cost = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             data, beta, lambda, gpu, rhoStar, weighting, targetact);
        end
    else
        batchsize = 5000;
        l = size(data,2);
        numbatches = ceil(l/ batchsize);
        if nargout > 1
            cost = 0;
            grad = 0;
            for i=1:numbatches
                batchdata = data(:, (i-1)*batchsize + 1:min(i*batchsize, l));
                [batchcost,batchgrad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                    batchdata, beta, lambda, gpu, rhoStar, weighting, targetact);
                cost = cost + batchcost;
                grad = grad + batchgrad;
            end
            cost = cost/numbatches;
            grad = grad/numbatches;
        else
            cost = 0;
            for i=1:numbatches
                batchdata = data(:, (i-1)*batchsize + 1:min(i*batchsize, l));
                batchcost = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                    batchdata, beta, lambda, gpu, rhoStar, weighting, targetact);
                cost = cost + batchcost;
            end
            cost = cost/numbatches;
        end
    end
end
end

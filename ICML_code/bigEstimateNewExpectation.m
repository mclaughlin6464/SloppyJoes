function q2 = bigEstimateNewExpectation(oldtheta, visibleSize, hiddenSize, ...
                                             data, beta, lambda, gpu, rhoStar, weighting, compressedHiddenSize, indices, indexVector)
if gpu == 1 && size(data,2) > 5000
    batchsize = 5000;
    l = size(data,2);
    numbatches = ceil(l/ batchsize);
else
    numbatches = 1;
    batchsize = size(data,2);
end
meanH = 0;
for i = 1:numbatches
    if i*batchsize > size(data,2)
        batchdata = data(:,(i-1)*batchsize+1 : size(data,2));
    else
        batchdata = data(:,(i-1)*batchsize+1 : i*batchsize);
    end
    meanH = meanH + estimateNewExpectation(oldtheta, visibleSize, hiddenSize, ...
                                             batchdata, beta, lambda, gpu, rhoStar, weighting, compressedHiddenSize, indices, indexVector);
end
meanH = meanH/numbatches;
q = weighting * meanH;
clear meanH;
q2 = q + (1 - weighting)*rhoStar;
q2 = double(q2);
end 
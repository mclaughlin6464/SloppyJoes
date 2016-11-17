function meanH = estimateNewExpectation(oldtheta, visibleSize, hiddenSize, ...
                                             data, beta, lambda, gpu, rhoStar, weighting, compressedHiddenSize, indices, indexVector)
global usegpu;
theta = expandParams(oldtheta, compressedHiddenSize, indices, indexVector);
if ~exist('gpu','var')
    gpu = 0;
    usegpu = 0;
else
    if strcmp(usegpu, 'gpumat') == 0
        usegpu = 'jacket';
    end
end

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
b1 = theta(hiddenSize*visibleSize+1:hiddenSize*visibleSize+hiddenSize);

if gpu == 1
    if strcmp(usegpu, 'gpumat') == 1
        W1 = gpusingletype1(W1);
        b1 = gpusingletype1(b1);
        data = gpusingletype1(data);
    else
        W1 = gsingle(W1);
        b1 = gsingle(b1);
        data = gsingle(data);
    end
end
% Forward Prop
h = sigmoid(bsxfunwrap(@plus, W1*data, b1));
clear W1 b1 data;
meanH = mean(h,2);
clear h;
end                 
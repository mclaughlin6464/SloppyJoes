function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             data, beta, lambda, gpu, rhoStar, weighting, targetact)
%This function contains the code for the objective function in the model.
%
% INPUT:
% theta                 -       current estimate of the parameter vector
%                               for the model.
% visibleSize           -       the number of input units
% hiddenSize            -       the number of hidden units
% lambda                -       weight decay parameter
% sparsityParam         -       The desired average activation for the hidden units 
% beta                  -       weight of sparsity penalty term
% data                  -       Our matrix containing the training data.  
%                               So, data(:,i) is the i-th training example.
% gpu                   -       1 to use gpu, 0 to use cpu.
% rhoStar               -       current estimate of the expected activation
%                               of the hidden neurons.
% weighting             -       the weight of the update term to the
%                               expected activation of the hidden neurons.
% targetact             -       the target activation value.
%
% OUPTUT:
% cost                  -       the value of the objective function.
% grad                  -       the gradient vector of the objective
%                               function at the current value of theta.
%
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes.
% This code is optimized to clear gpu data from the gpu memory as soon as
% its purpose is fulfilled.
global usegpu;
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
b2 = theta(hiddenSize*visibleSize+hiddenSize+1:end);
%Implementing weight tying
W2 = W1';
if gpu == 1
    if strcmp(usegpu, 'gpumat') == 1
        W1 = gpusingletype1(W1);
        W2 = gpusingletype1(W2);
        b1 = gpusingletype1(b1);
        b2 = gpusingletype1(b2);
        data = gpusingletype1(data);
        rhoStar = gpusingletype1(rhoStar);
    else
        W1 = gsingle(W1);
        W2 = gsingle(W2);
        b1 = gsingle(b1);
        b2 = gsingle(b2);
        data = gsingle(data);
        rhoStar = gsingle(rhoStar);
    end
end
W1grad = 0;
W2grad = 0;
b1grad = 0;
b2grad = 0;




cost = 0;
M = size(data, 2);

% Forward Prop
h = sigmoid(bsxfunwrap(@plus, W1*data, b1));
r = bsxfunwrap(@plus, W2*h, b2);

if beta ~= 0
    % Sparsity Cost
    p = targetact; %sparsityParam;
    q = weighting * mean(h, 2);
    q2 = q + (1 - weighting)*rhoStar;
    cost = sum(beta * ( p*log(p./q2) + (1-p)*log((1-p)./(1-q2))));
end
% Reconstruction Loss and Back Prop
diff = r - data;
clear r;
cost = cost + 1/M * 0.5 * sum(diff(:).^2);

if lambda ~=0
    cost = cost + sum(lambda * 0.5 * (W1(:).^2)) ...
            + sum(lambda * 0.5 * (W2(:).^2));
end
cost = double(cost);
if nargout > 1
    
    outderv = 1/M * diff;
    clear diff;
    % Output Layer
    W2grad = outderv * h';
    b2grad = sum(outderv, 2);
    outderv = W2' * outderv;

    if beta ~= 0
            outderv = outderv + 1/M * repmat(beta * (-p./q2 + (1-p) ./ (1-q2)), 1, M) * weighting;
    end
    clear p q2 q;
    % Hidden Layer
    outderv = outderv .* h .* (1 - h);
    clear h;
    W1grad = outderv * data';
    clear data;
    b1grad = sum(outderv, 2);
    clear outderv;
    if lambda ~=0 
        % Weight Regularization
        W1grad = W1grad + lambda * W1;
        W2grad = W2grad + lambda * W2;
        % ---------- End Sample Solution ----------
    end
    
    
    
    %-------------------------------------------------------------------
    % After computing the cost and gradient, we will convert the gradients back
    % to a vector format (suitable for minFunc).  Specifically, we will unroll
    % your gradient matrices into a vector.
    W1grad = W1grad + W2grad';    
    grad = [W1grad(:); b1grad(:) ; b2grad(:)];
    grad = double(grad);
    clear W1 W2 b1 b2 W1grad W2grad b1grad b2grad;
end
end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end


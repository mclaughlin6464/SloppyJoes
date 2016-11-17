function [theta indices indexVector hiddenSize compressedHiddenSize] = initializeLocalrfParameters(nummaps, visibleSize, windowSize, step, outside)
% This function initializes the parameters for the localrf autoencoder.
% INPUT:
% nummaps               -           number of feature maps you want to learn
% visibleSize           -           the size of the input layer.
% windowSize            -           the size of the receptive field.
% step                  -           the stride size (how many pixels we can skip), 
%                                   default is 1
% outside               -           1 allows the receptive fields to see 
%                                   only a small section of
%                                   the image when at the corners (so that 
%                                   we can have number of receptive fields 
%                                   = number of pixels). outside = 0 will 
%                                   force all receptive fields to
%                                   have the same side
% OUTPUT:
% theta                 -           the initialized localrf parameter vector  
% indices               -           the indices matrix.
% hiddenSize            -           the size of the hidden layer.
% compressedHiddenSize  -           the length of the weight vector into
%                                   the hidden layer not counting the
%                                   biases.
imageWidth = sqrt(visibleSize);
imageHeight = imageWidth;
indices = createIndicesMatrix(imageWidth, imageHeight, windowSize, step, outside);
indices = repmat(indices, nummaps, 1);
hiddenSize = size(indices, 1);

%% Initialize parameters randomly based on layer sizes.
r  = sqrt(6) / sqrt(hiddenSize+visibleSize);   % we'll choose weights uniformly from the interval [-r, r]
W1 = rand(hiddenSize, visibleSize)*2*r - r;
b1 = zeros(hiddenSize, 1);
b2 = zeros(visibleSize, 1);

% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 

indices = logical(indices);
indexVector = createIndexVector(indices);
W1 = collapseWeightsFull(W1, indices, indexVector);
compressedHiddenSize = length(W1(:));
theta = [W1(:) ; b1(:) ; b2(:)];

end


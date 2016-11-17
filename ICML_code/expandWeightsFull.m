function W2 = expandWeightsFull(Wsmall, indices, indexVector)
 W2 = zeros(size(indices));
 A = Wsmall';
 W2(indexVector) = A(:)';
 W2 = reshape(W2, size(W2,2), size(W2,1))';
end
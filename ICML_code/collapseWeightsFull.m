function Wsmallnew2 = collapseWeightsFull(W, indices, indexVector)
n = size(W(1,indices(1,:)),2);
m = size(W,1); 
A = W';
Wsmallnew2 = reshape(A(indexVector), n, m)';
end
function [] = slaveLoadDataLocalrf(batchStart, batchEnd, batchsize, indices, indexVector, compressedHiddenSize)
[~, ~, data] = makebatches(1000);
global state; 
fprintf('batchStart = %d, batchEnd = %d, batchsize = %d\n', batchStart, batchEnd, batchsize);
data = data';
data = double(data(:,batchStart:batchEnd));
m = size(data,2);
numbatches = m / batchsize;
state.data = zeros(size(data,1), batchsize, numbatches);
for i=1:numbatches
    state.data(:,:,i) = data(:, (i-1)*batchsize + 1:i*batchsize);
end
state.numbatches = numbatches;
state.indices = indices;
state.indexVector = indexVector;
state.compressedHiddenSize = compressedHiddenSize;
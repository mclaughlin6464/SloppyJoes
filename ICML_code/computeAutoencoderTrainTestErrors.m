function [trainerr, testerr] = computeAutoencoderTrainTestErrors(params, hiddenSize, visibleSize, batchdata, testbatchdata, beta, lambda, targetact, compressedHiddenSize, indices, indexVector)
err=0;
%This function computes the value of the objective function on the train
%and test datsets.
%%%%%%%%%%%%%%%%%%%% COMPUTE OBJECTIVE VALUE ON TRAIN SET %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[numcases numdims numbatches]=size(batchdata);
newparams = expandParams(params, compressedHiddenSize, indices, indexVector);
N=numcases;
for batch = 1:numbatches
    data = [batchdata(:,:,batch)];
    data = data';
    err = err + sparseAutoencoderCost(newparams, visibleSize, hiddenSize,  ...
                                             data, beta, lambda, 0, 0, 1, targetact);
end
trainerr = err/numbatches;

%%%%%%%%%%%%%%%%%%%% COMPUTE OBJECTIVE VALUE ON TEST SET %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[testnumcases testnumdims testnumbatches]=size(testbatchdata);
N=testnumcases;
err=0;
for batch = 1:testnumbatches
    data = [testbatchdata(:,:,batch)];
    data = data';
    err = err + sparseAutoencoderCost(newparams, visibleSize, hiddenSize, ...
                                             data, beta, lambda, 0, 0, 1, targetact);
end

testerr=err/testnumbatches;

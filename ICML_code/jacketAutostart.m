function [] = jacketAutostart

%% Auto Start Jacket on a Free GPU
global usegpu selgpu gpusingletype singletype;
addpath /usr/local/jacket/engine/

if usegpu
    disp('Warning: GPU already in use! Not initializing');
    return
end

% gactivate;

usegpu = 'jacket';
selgpu = 0;

gpusingletype = @gsingle;
singletype    = @gsingle;

freemem = getfield(gpu_entry(13), 'gpu_free');

[idum,hostname]= system('hostname');

if strncmp(hostname, 'gryphon', 7)
    % on a gryphon
    if (freemem < 1.02e+09)
        disp('Warning: The GPU seems really busy!');
    end
elseif strncmp(hostname, 'gorgon', 6) 
    % on a gorgon
    if (freemem < 1.135e+09)
        gselect(1);
        selgpu = 1;
        freemem = getfield(gpu_entry(13), 'gpu_free');
        if (freemem < 1.135e+09)
            disp('Warning: Both GPUs seem really busy!');
        end
    end
else
    disp(['Warning: Host Not Recognized -- ' hostname]);
end
disp(['Selected GPU: ' num2str(selgpu)]);

% create a dummy variable to use up some memory
tmp = gsingle(0); gsync;




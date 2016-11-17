function indices = createIndicesMatrix(imageWidth, imageHeight, windowSize, step, outside)
% creates an indices matrix that defines the location of each receptive
% field
%
% INPUT:
% imageWidth            -           the widht of the image
% windowSize            -           the size of the receptive field
% step                  -           the stride size (how many pixels we can skip), 
%                                   default is 1
% outside               -           1 allows the receptive fields to see 
%                                   only a small section of
%                                   the image when at the corners (so that 
%                                   we can have number of receptive fields 
%                                   = number of pixels). outside = 0 will 
%                                   force all receptive field to
%                                   have the same side
%
% OUTPUT:
% indices               -           the indice matrix created.

if nargin < 5
    outside = 0;
end
if nargin < 4
    step = 1;
end
%fprintf('imageWidth = %d, imageHeight = %f, windowSize = %d, step = %d, outside = %d\n', imageWidth, imageHeight, windowSize, step, outside);
% allow the receptive field to stick outside of the image or not?
if outside == 0
    windowWidth = windowSize;
    windowHeight = windowSize;
    counter = 1;
%     fprintf('imageHeight-windowHeight = %d\n', imageHeight-windowHeight);
    assert(~isempty(1:step:imageHeight-windowHeight+1), 'step is too large');
    indices = zeros(floor((imageHeight-windowHeight+1)/step) *floor((imageWidth-windowWidth+1)/step), imageWidth*imageHeight);
    
    for startRow=1:step:imageHeight-windowHeight+1
%         fprintf('row = %d out of %d\n', startRow, imageHeight-windowHeight+1);
        for startCol=1:step:imageWidth-windowWidth+1
            tempIndices = zeros(imageHeight, imageWidth);
            tempIndices(startRow:startRow+windowHeight-1,startCol:startCol+windowWidth-1) = 1;
            tempIndices = reshape(tempIndices, 1, imageWidth*imageHeight);
            
            indices(counter,:) = tempIndices;
            counter = counter + 1;
        end
    end
    assert(size(indices,1)==counter-1);
else
    bigImageHeight = imageHeight + windowSize;
    bigImageWidth = imageWidth + windowSize;
    windowWidth = windowSize;
    windowHeight = windowSize;
    counter = 1;
    indices = zeros(floor((bigImageHeight-windowHeight+1)/step) *floor((bigImageWidth-windowWidth+1)/step), imageWidth*imageHeight);
    for startRow=1:step:bigImageHeight-windowHeight+1
        fprintf('row = %d out of %d\n', startRow, bigImageHeight-windowHeight+1);
        for startCol=1:step:bigImageWidth-windowWidth+1
            tempIndices = zeros(bigImageHeight, bigImageWidth);
            tempIndices(startRow:startRow+windowHeight-1,startCol:startCol+windowWidth-1) = 1;
            smallTempIndices = tempIndices(windowSize/2+1:windowSize/2+imageHeight, windowSize/2+1:windowSize/2+imageWidth);
            smallTempIndices = reshape(smallTempIndices, 1, imageWidth*imageHeight);
            
            indices(counter,:) = smallTempIndices;
            counter = counter + 1;
        end
    end
    assert(size(indices,1)==counter-1);
end


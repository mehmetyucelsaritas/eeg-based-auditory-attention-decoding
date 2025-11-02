function newS = fft_resample(oldS, oldFs, newFs)
% function newS = fft_resample(oldS, oldFs, newFs)
% 
% Resample the columns of a 2D matrix, one row at a time. The input parameter
% oldS is the array of data, one signal per column, oldFs is the original
% (old) sample rate, newFs is the new (desired) sample rate.

% malcolm@ieee.org   July 3, 2015

newS = [];
for col=1:size(oldS,2)
    newPiece = fft_resample_1d(oldS(:,col), oldFs, newFs);
    if isempty(newS)
        newS = zeros(size(newPiece,1), size(oldS,2));
    end
    newS(:,col) = newPiece;
end

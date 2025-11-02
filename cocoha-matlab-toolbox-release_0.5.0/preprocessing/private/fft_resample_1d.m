function newS = fft_resample_1d(oldS, oldFs, newFs)
% function newS = fft_resample(oldS, oldFs, newFs)
%
% Resample an (old) signal to a new sample rate.  Original (old) sample
% rate is oldFs, new sample rate is newFs. This code only processes
% one-dimensional signals, 1xN samples in size.
%
% This code uses an FFT to perform the interpolation.  When the sample rate
% is reduced, reduce the size of the size of the FFT by cutting off the
% high frequency components.  When the sample rate is increased, zero pad
% the FFT to extend the signal.

if nargin < 1
    oldFs = 10;
    oldS = sin(0:1/oldFs:9.999)';
    newFs = 5;              % 5 and 20 are good values to try
    newS = fft_resample_1d(oldS, oldFs, newFs);
    oldT = (0:length(oldS)-1)'/oldFs;
    newT = (0:length(newS)-1)'/newFs;
    
    handles = plot(oldT, oldS, oldT, oldS, 'o', ...
        newT, newS, newT, newS, 'x');
    xlabel('Time ->');
    legend(handles([1 3]), 'Original', 'Resampled');
    return;
end
extendedS = [oldS; 0*oldS];         % zero pad the data.

extN = length(extendedS);
extN2 = extN/2;
extN2p = extN - extN2;

newN = round(extN*newFs/oldFs);
newN2 = floor(newN/2);
newN2p = newN2;  % newN - newN2;

fOld = fft(extendedS);

fNew = zeros(newN, 1);
if newFs > oldFs
    % Upsample: Need to add zeros in the middle
    fNew(1:extN2) = fOld(1:extN2);
    fNew(end-extN2p+1:end) = fOld(end-extN2p+1:end);
else
    % Downsample: Need to keep LF coefficients
    fNew(1:newN2) = fOld(1:newN2);
    fNew(end-newN2p+1:end) = fOld(end-newN2p+1:end);
end
newS = ifft(fNew)*newFs/oldFs;
newS = real(newS(1:newN2));
if sum(isnan(newS)) > 0
    error('Got a NaN in fft_resample_1d');
end


    
%% Figure Related Functions
function [fig, startIdx, maxIndex, windowSize] = initSlidingWindowFigure(eegMatrix)
    windowSize = 10000;        % Window size (# of samples visible at once)
    N = size(eegMatrix, 1);    % Length of EEG signal (assumes sample vector or matrix)
    maxIndex = N - windowSize; % Maximum slider index
    startIdx = 1;              % Initial index
    fig = figure('Name', 'Sliding Window Plot'); % Create figure for sliding window display
end
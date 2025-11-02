% This script demonstrates how to import and align EEG and AUDIO data using
% the COCOHA Matlab Toolbox v0.5.0, found here: http://doi.org/10.5281/zenodo.1198430

EEGBASEPATH = './eeg';           % Find EEG files here
WAVBASEPATH = './audio';         % Find AUDIO wav files here 
MATBASEPATH = '.';               % Save preprocessed data files here
addpath(genpath("cocoha-matlab-toolbox-release_0.5.0/"))

for ss = 1:18
    clear data data_noise
    fprintf('Processing subject: %s\n', num2str(ss));
    
    % Load data from original .mat file
    % load(fullfile(EEGBASEPATH,['S' num2str(ss) '.mat']))

    % Save data with only with one channel eeg signal
    % data.eeg{1} = data.eeg{1}(1:500000, :);
    % save("output/data.mat", "data", "expinfo");

    % Load data from saved one channel instance data for fast processes
    load("output/data.mat")

    %% Figure related initializations
    % Window size
    windowSize = 10000; % number of points to show at once
    N = length(data.eeg{1}(:, 1));
    maxIndex = N - windowSize;
    
    % Initial start index
    startIdx = 1;
    
    % Create figure
    fig = figure('Name','Sliding Window Plot');
    
    % customPlot(data.eeg{1}(:, 1), 'Raw Data');

    h1 = subplot(4,1,1);
    raw_data = data.eeg{1}(:, 1);
    p1 = plot(raw_data);
    title('Raw Data');

    data.cfg = [];
    
    %% Line noise filtering 50 Hz
    cfg = [];
    cfg.eeg.smooth = data.fsample.eeg/50;
    data = co_preprocessing(cfg,data);
    
    % customPlot(data.eeg{1}(:, 1), '50Hz Noise eleminated Data');
    h2 = subplot(4,1,2);
    data50HzNoiseRemoved = data.eeg{1}(:, 1);
    p2 = plot(data50HzNoiseRemoved);
    title('50Hz Noise eleminated Data');

    %% Downsample
    % cfg = [];
    % cfg.eeg.newfs = 64;
    % data = co_resampledata(cfg,data);

    % customPlot(data.eeg{1}(1000:50000, 1), 'Downsampled Data');


    %% Initial filtering
    cfg = [];
    cfg.eeg.detrend = 1;    
    cfg.eeg.hpfilter = 'yes';
    cfg.eeg.hpfilttype = 'butter'; 
    cfg.eeg.hpfiltord = 2; 
    cfg.eeg.hpfiltdir = 'onepass';
    cfg.eeg.hpfreq = 0.1;
    data = co_preprocessing(cfg,data);

    % customPlot(data.eeg{1}(:, 1), 'Initial Filtered Data');

    h3 = subplot(4,1,3);
    filtereData = data.eeg{1}(:, 1);
    p3 = plot(filtereData);
    title('Initial Filtered Data');


    %% Remove original EOG and unused channels from data and average reference
    cfg = [];
    cfg.eeg.channels = {'all','-EXG3','-EXG4','-EXG5','-EXG6','-EXG7','-EXG8', '-EXG1', '-EXG2','-Status'};
    cfg.eeg.reref = 'yes';
    cfg.eeg.refchannel = 'all';
    data = co_preprocessing(cfg,data);

    %% Average reference
    cfg = [];
    cfg.eeg.reref = 'yes';
    cfg.eeg.refchannel = 'all';
    data = co_preprocessing(cfg,data);
    
    % customPlot(data.eeg{1}(:, 1), 'Average referenced Data');
    h4 = subplot(4,1,4);
    averagedData = data.eeg{1}(:, 1);
    p4 = plot(averagedData);
    title('Average referenced Data');

    slider = uicontrol('Style', 'slider',...
    'Min',1,'Max',maxIndex,'Value',startIdx, ...
    'Units','normalized','Position',[0.25 0.01 0.5 0.03]);

    % Slider callback function
    slider.Callback = @(src,event) updatePlots(round(src.Value), windowSize, raw_data, data50HzNoiseRemoved, filtereData, averagedData, p1, p2, p3, p4);
    a = 3;
    break
end

function customPlot(dataSnippet, text)
    subplot(4,1,1);
    figure;
    plot(dataSnippet);
    title(text);
end

% Function to update plots
function updatePlots(startIdx, windowSize, raw_data, data50HzNoiseRemoved, filtereData, averagedData, p1, p2, p3, p4)
    idx = startIdx:(startIdx+windowSize-1);
    p1.YData = raw_data(idx);
    p1.XData = idx;
    xlim(p1.Parent, [startIdx, startIdx+windowSize-1]);

    p2.YData = data50HzNoiseRemoved(idx);
    p2.XData = idx;
    xlim(p2.Parent, [startIdx, startIdx+windowSize-1]);

    p3.YData = filtereData(idx);
    p3.XData = idx;
    xlim(p3.Parent, [startIdx, startIdx+windowSize-1]);

    p4.YData = averagedData(idx);
    p4.XData = idx;
    xlim(p4.Parent, [startIdx, startIdx+windowSize-1]);

    drawnow;
end

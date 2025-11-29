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

    h1 = subplot(5,1,1);
    raw_data = data.eeg{1}(:, 1);
    p1 = plot(raw_data);
    title('Raw Data');

    data.cfg = [];
    
    %% Line noise filtering 50 Hz
    cfg = [];
    cfg.eeg.smooth = data.fsample.eeg/50;
    data = co_preprocessing(cfg,data);
    
    % customPlot(data.eeg{1}(:, 1), '50Hz Noise eleminated Data');
    h2 = subplot(5,1,2);
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

    h3 = subplot(5,1,3);
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
    h4 = subplot(5,1,4);
    averagedData = data.eeg{1}(:, 1);
    mean(averagedData)
    p4 = plot(averagedData);
    title('Average referenced Data');
     
    %% Threshold-based artifact removal
    % h5 = subplot(5,1,5);
    % cleanData = removeArtifactsThreshold(data.eeg{1}(:, 1), 40, 200);
    % p5 = plot(cleanData);
    % title('Threshold-based artifact removal');

    %% Ica base artifact removal
    % EEG = convertToEEGLAB(data, 100000);
    % EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1);
    % EEG = pop_chanedit(EEG, 'lookup', 'standard-10-5-cap385.elp');
    load('output/ICAweights.mat', 'EEG');
    EEG = pop_subcomp(EEG, [1 4 5 8 17 34 35 36 45 46 52 57], 0);
    
    %% Asr based automatic artifat removal
    calibrationData = EEG.data(1:28, 1:30720);
    samplingRate = EEG.srate;
    % Compute ASR calibration state
    cutoff_scalar = 20;
    asr_state = asr_calibrate(calibrationData, samplingRate, cutoff_scalar);
    EEG_ASR = asr_process(EEG.data(1:28, :), samplingRate, asr_state);

    h5 = subplot(5,1,5);
    p5 = plot(EEG_ASR(1, :));
    title('Asr-based artifact removal');

    %% Slider configurations
    slider = uicontrol('Style', 'slider',...
    'Min',1,'Max',maxIndex,'Value',startIdx, ...
    'Units','normalized','Position',[0.25 0.01 0.5 0.03]);
    linkaxes([h1, h2, h3, h4, h5], 'x'); 
    % Slider callback function
    slider.Callback = @(src,event) updatePlots(round(src.Value), windowSize, ...
                    raw_data, data50HzNoiseRemoved, filtereData, averagedData, EEG_ASR(1, :), ...
                    p1, p2, p3, p4, p5);
    a = 3;
    break
end

function customPlot(dataSnippet, text)
    subplot(5,1,1);
    figure;
    plot(dataSnippet);
    title(text);
end

% Function to update plots
function updatePlots(startIdx, windowSize, raw_data, data50HzNoiseRemoved, filtereData, averagedData, cleanData, p1, p2, p3, p4, p5)
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

    p5.YData = cleanData(idx);
    p5.XData = idx;
    xlim(p5.Parent, [startIdx, startIdx+windowSize-1]);
    ylim(p5.Parent, [-100, 300]);

    drawnow;
end

function [cleanEEG] = removeArtifactsThreshold(eegData, threshold, windowSize)

    nSamples = length(eegData);
    for startIdx = 1:windowSize:nSamples
        endIdx = min(startIdx + windowSize - 1, nSamples);
        window = eegData(startIdx:endIdx);
        % Check if any value exceeds threshold
        if any(abs(window) > threshold)
            artifact_idx = window > threshold;
            window(artifact_idx) = NaN;
            % Truncate the artifact (here set to NaN)
            eegData(startIdx:endIdx) = window;
        end
    end
    % Find samples exceeding threshold in any channel
    cleanEEG = fillmissing(eegData, 'linear');
end

function EEG = convertToEEGLAB(data, nDataPoint)

    % Create empty EEG structure
    EEG = [];

    % ---------------------------------------------------------
    % Basic fields
    % ---------------------------------------------------------
    EEG.data   = data.eeg{1}(1:nDataPoint, :)';          % EEGLAB expects chan x samples
    EEG.srate  = data.fsample.eeg;        % Sampling rate
    EEG.nbchan = size(EEG.data, 1);    % Number of channels
    EEG.pnts   = size(EEG.data, 2);    % Number of samples
    EEG.trials = 1;                   % Continuous recording
    EEG.xmin   = 0;
    EEG.xmax   = (EEG.pnts-1)/EEG.srate;

    % ---------------------------------------------------------
    % Channel locations (labels only, no coordinates yet)
    % ---------------------------------------------------------
    labels = data.dim.chan.eeg{1};       % 64×1 cell
    nChan = length(labels(:, :));     
    for i = 1:nChan
        EEG.chanlocs(i).labels = labels{i};
    end

    % ---------------------------------------------------------
    % Events
    % ---------------------------------------------------------
    % if isfield(data, 'event') && ~isempty(data.event)
    %     EEG.event = data.event;       % Should work if event format is simple
    % else
    %     EEG.event = [];
    % end

    % ---------------------------------------------------------
    % Empty ICA fields
    % ---------------------------------------------------------
    EEG.icaact   = [];
    EEG.icaweights = [];
    EEG.icasphere = [];
    EEG.icachansind = [];

    % ---------------------------------------------------------
    % Internal EEGLAB bookkeeping
    % ---------------------------------------------------------
    EEG.setname = 'myEEG'; 
    EEG.icawinv = [];
    EEG.filename = 'myEEG.set';
    EEG.filepath = pwd;   
    EEG = eeg_checkset(EEG);

end

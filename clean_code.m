% This script demonstrates how to import and align EEG and AUDIO data using
% the COCOHA Matlab Toolbox v0.5.0, found here: http://doi.org/10.5281/zenodo.1198430
% EEGLAB Toolbox v2025.1.0, found here: https://eeglab.org/

% Load data from saved one channel instance data for fast processes
load("output/data.mat");
data.cfg = [];
analysisComponent = false; % true: to analysis components, false: use ICA weights
manualCalibration = true; % true: manual, false: ICA ouput

%% Figure related initializations
[fig, startIdx, maxIndex, windowSize] = initSlidingWindowFigure(data.eeg{1});

%% Noisy Raw Data
raw_data = data.eeg{1}(:, 1);

%% Line noise filtering 50 Hz
data = co_preprocessing(lineNoiseFilter50HzConfig(data), data);
data50HzNoiseRemoved = data.eeg{1}(:, 1);

%% Downsample
% data = co_resampledata(downsampleConfig(64), data);
% dataDownsamlped = data.eeg{1}(:, 1);

%% Initial High Pass filtering
data = co_preprocessing(highPassFilterConfig(), data);
filteredData = data.eeg{1}(:, 1);

%% Remove original EOG and unused channels from data and average reference
data = co_preprocessing(rereferenceConfig(), data);
averagedData = data.eeg{1}(:, 1);

%% Threshold-based artifact removal
% cleanData = removeArtifactsThreshold(data.eeg{1}(:, 1), 40, 200);

%% Ica base artifact removal
if analysisComponent
    EEG = convertToEEGLAB(data, 100000);
    EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1);
    EEG = pop_chanedit(EEG, 'lookup', 'standard-10-5-cap385.elp');
    pop_topoplot(EEG, 0, 1:EEG.nbchan, 'ICA Component Scalp Maps', 0, 'electrodes', 'on');
else
    load('output/ICAweights.mat', 'EEG');
    EEG = pop_subcomp(EEG, [1 4 5 8 17 34 35 36 45 46 52 57], 0); 
end

%% Asr based artifact removal
calibDataSize = EEG.srate*2;
calibChanSize = EEG.nbchan;
calibrationData = EEG.data(1:calibChanSize, 1:calibDataSize);
samplingRate = EEG.srate;

% Compute ASR calibration state
cutoff_scalar = 20;
if manualCalibration
    load('output/calibrationDataManual.mat', 'calibrationDataManual');
    asr_state = asr_calibrate(calibrationDataManual, samplingRate, cutoff_scalar); %calibration based on manually extracted output
else
    asr_state = asr_calibrate(calibrationData, samplingRate, cutoff_scalar); %calibration based on ICA output 
end 
EEG_ASR = asr_process(EEG.data(1:calibChanSize, :), samplingRate, asr_state);

%% Initialize h:subplot axis handle and p:plot handle.
titles = {'Raw Data', '50Hz Noise eliminated Data',...
    'Initial Filtered Data', 'Average referenced Data', ...
    'ICA-cleaned Data', 'Asr-based artifact removal'};

dataList = {raw_data, data50HzNoiseRemoved, filteredData, ...
    averagedData, EEG.data(1,:)', EEG_ASR(1, :)};

for i = 1:length(dataList)
    [h{i}, p{i}] = createSubplotWithPlot(i, 6, dataList{i}, titles{i});
end

%% Slider configurations
slider = uicontrol('Style', 'slider', 'Min', 1, 'Max', maxIndex, 'Value', startIdx, ...
'Units','normalized','Position',[0.25 0.01 0.5 0.03]);
linkaxes([h{:}], 'x'); 

% Slider callback function
slider.Callback = @(src,event) updatePlots(round(src.Value), windowSize, dataList, p, titles);

%% Figure Related Functions
function [fig, startIdx, maxIndex, windowSize] = initSlidingWindowFigure(eegMatrix)
    windowSize = 10000;        % Window size (# of samples visible at once)
    N = size(eegMatrix, 1);    % Length of EEG signal (assumes sample vector or matrix)
    maxIndex = N - windowSize; % Maximum slider index
    startIdx = 1;              % Initial index
    fig = figure('Name', 'Sliding Window Plot'); % Create figure for sliding window display
end

function [h, p] = createSubplotWithPlot(counter, totalRows, dataVector, plotTitle)
    % Automatically create the next subplot row
    h = subplot(totalRows, 1, counter);
    p = plot(dataVector);
    % Formatting
    title(plotTitle);
end

function updatePlots(startIdx, windowSize, dataCells, plotHandles, plotTitle)

    idx = startIdx:(startIdx+windowSize-1);
    
    for i = 1:length(plotHandles)
        p = plotHandles{i};
        dataVec = dataCells{i};
        
        % Update Y and X values
        p.YData = dataVec(idx);
        p.XData = idx;

        % Update axis limits
        xlim(p.Parent, [startIdx, startIdx + windowSize - 1]);

        % Only some plots need fixed y-limits
        if plotTitle{i} == "ICA-cleaned Data" || plotTitle{i} == "Asr-based artifact removal" % ICA and ASR
            ylim(p.Parent, [-100, 300]);
        end
    end
    drawnow;
end

%% Configuration related Functions
function cfg = lineNoiseFilter50HzConfig(data)
    cfg = [];
    cfg.eeg.smooth = data.fsample.eeg/50;
end

function cfg = highPassFilterConfig()
    % Configuration of High Pass Filtering
    cfg = [];
    cfg.eeg.detrend     = 1;
    cfg.eeg.hpfilter    = 'yes';
    cfg.eeg.hpfilttype  = 'butter';
    cfg.eeg.hpfiltord   = 2;
    cfg.eeg.hpfiltdir   = 'onepass';
    cfg.eeg.hpfreq      = 0.1;
end

function cfg = downsampleConfig(newRate)
    cfg = [];
    cfg.eeg.newfs = newRate;
end

function cfg = rereferenceConfig()
    % Configure EEG re-referencing and channel selection
    cfg = [];
    cfg.eeg.channels = {'all','-EXG3','-EXG4','-EXG5','-EXG6','-EXG7','-EXG8', '-EXG1', '-EXG2','-Status'};
    cfg.eeg.reref = 'yes';
    cfg.eeg.refchannel = 'all';
end

%% ICA Related Functions
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

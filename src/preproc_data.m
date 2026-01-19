% This script demonstrates how to import and align EEG and AUDIO data using
% the COCOHA Matlab Toolbox v0.5.0, found here: http://doi.org/10.5281/zenodo.1198430
% EEGLAB Toolbox v2025.1.0, found here: https://eeglab.org/

% Load data from original DTU dataset .mat file
% load(fullfile(EEGBASEPATH,['S' num2str(ss) '.mat']))
% data.eeg{1} = data.eeg{1}(1:500000, :);
% save("output/data.mat", "data", "expinfo");

% Load data from cropped DTU dataset for the fast data processes
load("output/data.mat"); % data.mat = ./eeg/S1/data.eeg{1}(1:500000, :);
data.cfg = [];
analysisComponent = false; % true: to analysis components, false: use ICA weights
manualCalibration = true; % true: manual, false: ICA ouput

%% Figure related initializations
[fig, startIdx, maxIndex, windowSize] = initSlidingWindowFigure(data.eeg{1});

%% Noisy Raw Data - processed only on first channel
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
EEG = convertToEEGLAB(data, 100000);
if analysisComponent
    EEG_ICA = pop_chanedit(EEG, 'lookup', 'standard-10-5-cap385.elp');
    EEG_ICA = pop_runica(EEG_ICA, 'icatype', 'runica', 'extended', 1);
    pop_topoplot(EEG_ICA, 0, 1:EEG_ICA.nbchan, 'ICA Component Scalp Maps', 0, 'electrodes', 'on');
else
    load('output/ICAweights.mat', 'EEG_ICA');
    EEG_ICA = pop_subcomp(EEG_ICA, [6 15], 0); 
end

%% Asr based artifact removal
calibDataSize = EEG.srate*2;
calibChanSize = EEG.nbchan;
calibrationData = EEG_ICA.data(1:calibChanSize, 1:calibDataSize);
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
    averagedData, EEG_ICA.data(1,:)', EEG_ASR(1, :)};

for i = 1:length(dataList)
    [h{i}, p{i}] = createSubplotWithPlot(i, 6, dataList{i}, titles{i});
end

%% Slider configurations
slider = uicontrol('Style', 'slider', 'Min', 1, 'Max', maxIndex, 'Value', startIdx, ...
'Units','normalized','Position',[0.25 0.01 0.5 0.03]);
linkaxes([h{:}], 'x'); 

% Slider callback function
slider.Callback = @(src,event) updatePlots(round(src.Value), windowSize, dataList, p, titles);

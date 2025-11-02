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
    
    customPlot(data.eeg{1}(:, 1), 'Raw Data');

    data.cfg = [];
    
    %% Line noise filtering 50 Hz
    cfg = [];
    cfg.eeg.smooth = data.fsample.eeg/50;
    data = co_preprocessing(cfg,data);
    
    customPlot(data.eeg{1}(:, 1), '50Hz Noise eleminated Data');

    %% Downsample
    % cfg = [];
    % cfg.eeg.newfs = 64;
    % data = co_resampledata(cfg,data);
    % 
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

    customPlot(data.eeg{1}(:, 1), 'Initial Filtered Data');

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
    
    customPlot(data.eeg{1}(:, 1), 'Average referenced Data');
    a = 3;
end

function customPlot(dataSnippet, text)
    subplot(4,1,1);
    figure;
    plot(dataSnippet);
    title(text);
end



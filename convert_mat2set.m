% Load your .mat file
load('EEG/S1.mat');  % replace with your actual file name

% Extract EEG matrix and metadata
EEGdata   = data.eeg{1};             % transpose: EEGLAB expects [channels x samples]
EEGdata   = EEGdata'; 
srate     = data.fsample.eeg;          % sampling rate
chanlabels = data.dim.chan.eeg{1};  % channel names

EEGdata = EEGdata(1, :);
EEGdata = EEGdata - mean(EEGdata);
chanlabels = chanlabels(1);
% --- Create EEGLAB dataset structure ---
EEG = pop_importdata('dataformat', 'array', ...
                       'data', EEGdata, ...
                       'srate', srate, ...
                       'nbchan', size(EEGdata,1));

% Add channel labels
for ch = 1:length(chanlabels)
    EEG.chanlocs(ch).labels = chanlabels{ch};
end

% --- Add event information ---
if isfield(data, 'event') && isfield(data.event, 'eeg')
    nsamp = size(EEG.data, 2);
    evSamples = double(data.event.eeg.sample);
    evValues  = {data.event.eeg.value{:}};  % make sure these are cells
    for i = 1:length(evSamples)
        if evSamples(i) <= nsamp
            EEG.event(i).type   = evValues{i};
            EEG.event(i).latency = evSamples(i);
        end
    end
end

% Finalize dataset
EEG = eeg_checkset(EEG);

% Optional: save as .set file
EEG = pop_saveset(EEG, 'filename', 'converted_from_mat.set');
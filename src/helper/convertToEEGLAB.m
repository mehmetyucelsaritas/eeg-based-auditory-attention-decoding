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

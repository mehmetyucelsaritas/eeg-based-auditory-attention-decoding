function cfg = rereferenceConfig()
    % Configure EEG re-referencing and channel selection
    cfg = [];
    cfg.eeg.channels = {'all','-EXG3','-EXG4','-EXG5','-EXG6','-EXG7','-EXG8', '-EXG1', '-EXG2','-Status'};
    cfg.eeg.reref = 'yes';
    cfg.eeg.refchannel = 'all';
end
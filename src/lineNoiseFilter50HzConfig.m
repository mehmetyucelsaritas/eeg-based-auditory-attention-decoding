function cfg = lineNoiseFilter50HzConfig(data)
    cfg = [];
    cfg.eeg.smooth = data.fsample.eeg/50;
end
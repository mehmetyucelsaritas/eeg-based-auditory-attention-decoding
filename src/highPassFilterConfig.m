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
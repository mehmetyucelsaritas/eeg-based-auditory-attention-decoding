function data = co_denoise(cfg,data)
% CO_DENOISE performs some semi-automated steps to remove line noise, eye movement and muscle
% artifacts from the data.
%
% INPUTS:
% cfg.FIELD.cell            = ['all'] or array of cell number(s). Cells of data field to denoise.
%
%
% cfg.FIELD.layout          = FieldTrip sensor layout filename. This will be used to for plotting
%                             DSS components (see cfg.FIELD.linedssthresh and
%                             cfg.FIELD.eog.dssthresh options).
%
% cfg.FIELD.line.samples    = ['all'] or cells containing an array of time samples to be included in
%                             line noise sampling.
% cfg.FIELD.line.freq       = [50] line noise frequency in Hertz. In Europe, this is 50 Hz. In North
%                             America, this is 60 Hz.
% cfg.FIELD.line.nfft       = [512] number of frequency bins when performing FFT.
% cfg.FIELD.line.harm       = [3] number of harmonics to use. If any exceed the Nyquist frequency,
%                             they will be ignored.
% cfg.FIELD.line.dssthresh  = [0.8] fraction of maximum DSS component power to remove. If set to a 
%                             value below 0, a window will be presented to allow the user to select 
%                             the threshold with the left mouse button. The right mouse button
%                             allows the plotting of the DSS component just above the selected
%                             threshold level, provided FieldTrip is installed, the data has only
%                             'time' and 'chan' dimensions, and the 'chan' dimension has labels.
%
% cfg.FIELD.eog.channels    = Cell array of EOG channel names or array of channel indexes for
%                             performing eye blink artifact removal. This is done by first
%                             bandpassing the EOG channels (see cfg.FIELD.eog.bpfreq and
%                             cfg.FIELD.eog.bpfiltord), then performing a z-score threshold. Data
%                             segments that exceed this threshold are marked as eye blinks and used
%                             to generate the correlation matrix used by joint decorrelation (aka
%                             source separation).
% cfg.FIELD.eog.samples     = ['all'] or cells containing an array of time samples to be included in
%                             EOG detection (one cell per data cell).
% cfg.FIELD.eog.bpfreq      = [1 15] bandpass frequency for EOG artifact detection (FIRWS filter).
% cfg.FIELD.eog.bpfiltord   = bandpass filter order. By default this is equal to the data sample
%                             rate.
% cfg.FIELD.eog.zthresh     = [4] z-score threshold for automatic EOG artifact marking. Set to -1
%                             for interactive manual selection.
% cfg.FIELD.eog.dssthresh   = [0.8] fraction of maximum DSS component power to remove. If set to a 
%                             value below 0, a window will be presented to allow the user to select 
%                             the threshold with the left mouse button.  The right mouse button
%                             allows the plotting of the DSS component just above the selected
%                             threshold level, provided FieldTrip is installed, the data has only
%                             'time' and 'chan' dimensions, and the 'chan' dimension has labels.
% cfg.FIELD.eog.realtime    = if this field is present, matrices needed to regress out artifacts
%                             using the OVDSSArtifact.py OpenViBE python module will be stored here.
%
% cfg.FIELD.robustpca       = ['no']/'yes' perform robust PCA, which has been effective for removing
%                             sparse artifacts (e.g. muscle).
% cfg.FIELD.star            = performs sparse-time artifact removal when options are set.
%               .thresh     = [1] threshold for eccentricity measure.
%               .radius     = [Inf] maximum distance of neighboring electrodes.
%               .layout     = FieldTrip-style layout file (if radius is not Inf). Only channels in
%                             the specified layout file will be kept.
%               .depth      = [1] maximum number of channels to fix at each sample.
%               
%
% OUTPUTS:
% data                      = denoised data.
%
%
% See also: de Cheveigné, A. (2016). Sparse Time Artifact Removal. Journal of Neuroscience Methods, 262, 14-20.
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZ
% Author(s): Daniel D.E. Wong

% Add appropriate packages to the path
if ~exist('nt_dss0','file')
    addpath(fullfile(fileparts(which('co_defaults')), 'external', 'NoiseTools'));
    addpath(fullfile(fileparts(which('co_defaults')), 'external', 'NoiseTools', 'COMPAT'));
end
if ~exist('inexact_alm_rpca','file')
    addpath(fullfile(fileparts(which('co_defaults')), 'external', 'inexact_alm_rpca'));
    addpath(fullfile(fileparts(which('co_defaults')), 'external', 'inexact_alm_rpca', 'PROPACK'));
end

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);

for ii = 1:length(fields)
    % Set defaults
    if ~isfield(cfg.(fields{ii}),'cell'); cfg.(fields{ii}).cell = 'all'; end;
    if isfield(cfg.(fields{ii}),'line')
        if ~isfield(cfg.(fields{ii}).line,'samples'); cfg.(fields{ii}).line.samples = 'all'; end;
        if ~isfield(cfg.(fields{ii}).line,'freq'); cfg.(fields{ii}).line.freq = 50; end;
        if ~isfield(cfg.(fields{ii}).line,'nfft'); cfg.(fields{ii}).line.nfft = 512; end;
        if ~isfield(cfg.(fields{ii}).line,'harm'); cfg.(fields{ii}).line.harm = 3; end;
        if ~isfield(cfg.(fields{ii}).line,'dssthresh'); cfg.(fields{ii}).line.dssthresh = 0.8; end;
    end
    if ~isfield(cfg.(fields{ii}),'eog') || ~isfield(cfg.(fields{ii}).eog,'samples'); cfg.(fields{ii}).eog.samples = 'all'; end;
    if ~isfield(cfg.(fields{ii}),'eog') || ~isfield(cfg.(fields{ii}).eog,'bpfreq'); cfg.(fields{ii}).eog.bpfreq = [1 15]; end;
    if ~isfield(cfg.(fields{ii}),'eog') || ~isfield(cfg.(fields{ii}).eog,'bpfiltord'); cfg.(fields{ii}).eog.bpfiltord = data.fsample.(fields{ii}); end;
    if ~isfield(cfg.(fields{ii}),'eog') || ~isfield(cfg.(fields{ii}).eog,'zthresh'); cfg.(fields{ii}).eog.zthresh = 4; end;
    if ~isfield(cfg.(fields{ii}),'eog') || ~isfield(cfg.(fields{ii}).eog,'dssthresh'); cfg.(fields{ii}).eog.dssthresh = 0.8; end;
    if ~isfield(cfg.(fields{ii}),'robustpca'); cfg.(fields{ii}).robustpca = 'no'; end;
    
    % Assertions
    assert(any(strcmp('time',dim.(fields{ii}))), ['''time'' dimension does not exist for data.' fields{ii}]);
    assert(any(strcmp('chan',dim.(fields{ii}))), ['''chan'' dimension does not exist for data.' fields{ii}]);
    
    % Determine which cells to perform operations on
    if strcmp(cfg.(fields{ii}).cell,'all'); cells = 1:length(data.(fields{ii})); else cells = cfg.(fields{ii}).cell; end;
    
    % Make time dimension first
    timeix = find(strcmp('time',dim.(fields{ii})));
    cfgtmp = [];
    cfgtmp.(fields{ii}).shift = timeix-1;
    data = co_shiftdim(cfgtmp,data);
    
    % Line frequency removal
    if isfield(cfg.(fields{ii}),'line')
        fprintf('Performing line frequency removal on field: %s\n', fields{ii});
        
        if ischar(cfg.(fields{ii}).line.samples)
            samples = cell(1,length(data.(fields{ii})));
            for jj = 1:length(samples)
                samples{jj} = 1:size(data.(fields{ii}){jj},1);
            end
        else
            samples = cfg.(fields{ii}).line.samples;
        end
        
        % Ignore harmonic specifications greater than Nyquist frequency
        harm = 1:cfg.(fields{ii}).line.harm;
        harm(harm*cfg.(fields{ii}).line.freq > data.fsample.(fields{ii})/2) = [];
        assert(~isempty(harm),'Specified line noise frequency is greater than the Nyquist frequency.');
        
        for jj = cells
            % Make data 2D
            sz = size(data.(fields{ii}){jj});
            data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},sz(1),prod(sz(2:end)));

            [c0,c1]=nt_bias_fft(data.(fields{ii}){jj}(samples{jj},:), ...
                cfg.(fields{ii}).line.freq*harm/data.fsample.(fields{ii}), ...
                cfg.(fields{ii}).line.nfft);
            [todss,pwr0,pwr1]=nt_dss0(c0,c1);
            p1=pwr1./pwr0;
            todssinv = pinv(todss)';
            if cfg.(fields{ii}).line.dssthresh < 0
                h = figure; plot(p1./max(p1),'o'); title('Select Line Frequency Component Threshold');
                button = 3;
                while button ~= 1
                    figure(h);
                    [~,cfg.(fields{ii}).line.dssthresh,button] = ginput(1);
                    comp_no = find(p1/max(p1)>cfg.(fields{ii}).line.dssthresh); comp_no = comp_no(end);
                    if button==3 && ...
                            isfield(cfg.(fields{ii}),'layout') && ...
                            exist('ft_defaults','file') && ...
                            length(dim.(fields{ii})) == 2 && any(strcmp(dim.(fields{ii}),'chan')) && ...
                            isfield(data.dim,'chan') && isfield(data.dim.chan,fields{ii}) && ...
                            comp_no >= 1 && comp_no <= length(p1)
                        % Plot dss component
                        datadss = []; datadss.fsample = 1; datadss.time = 0;
                        datadss.dimord = 'chan_time';
                        datadss.label = data.dim.chan.(fields{ii}){jj};
                        datadss.avg = todssinv(:,comp_no);
                        ft_defaults;
                        cfgtmp = []; cfgtmp.parameter = 'avg'; cfg.zlim = 'maxabs';
                        cfgtmp.interactive = 'no'; cfgtmp.layout = cfg.(fields{ii}).layout;
                        cfgtmp.comment = ['DSS component ' num2str(comp_no)];
                        h_top=figure; ft_topoplotER(cfgtmp,datadss); uiwait(h_top);
                    end
                end
                close(h)
            end
            artix = find(p1/max(p1)>cfg.(fields{ii}).line.dssthresh);
            z=nt_mmat(data.(fields{ii}){jj},todss);
            data.(fields{ii}){jj} = nt_tsr(data.(fields{ii}){jj},z(:,artix));

            data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},sz);    % Restore dimensions
        end
    end
    
    % Eye blink removal
    if isfield(cfg.(fields{ii}).eog,'channels')
        fprintf('Performing eye blink removal on field: %s\n', fields{ii});
        
        if ischar(cfg.(fields{ii}).eog.samples)
            samples = cell(1,length(data.(fields{ii})));
            for jj = 1:length(samples)
                samples{jj} = 1:size(data.(fields{ii}){jj},1);
            end
        else
            samples = cfg.(fields{ii}).eog.samples;
        end
        
        % Preprocess EOG channels and find Z-score
        cfgtmp = [];
        cfgtmp.(fields{ii}).channels = cfg.(fields{ii}).eog.channels;
        cfgtmp.(fields{ii}).bpfilter = 'yes';
        cfgtmp.(fields{ii}).bpfilttype = 'firws';
        cfgtmp.(fields{ii}).bpfreq = cfg.(fields{ii}).eog.bpfreq; 
        cfgtmp.(fields{ii}).bpfiltord = cfg.(fields{ii}).eog.bpfiltord;
        dataeye = co_preprocessing(cfgtmp,data);
        for jj = cells
            cfgtmp = [];
            cfgtmp.(fields{ii}).cell = jj;
            cfgtmp.(fields{ii}).dim = 'time';
            cfgtmp.(fields{ii}).select = samples{jj};
            dataeye = co_selectdim(cfgtmp,dataeye);
        end
        cfgtmp = [];
        cfgtmp.(fields{ii}).zscore = 'yes';
        dataeye = co_preprocessing(cfgtmp,dataeye);

        for jj = cells
            % Make data 2D
            sz = size(data.(fields{ii}){jj});
            data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},sz(1),prod(sz(2:end)));
            
            szeye = size(dataeye.(fields{ii}){jj});
            zscore = max(abs(reshape(dataeye.(fields{ii}){jj},szeye(1),prod(szeye(2:end)))),[],2);
            if cfg.(fields{ii}).eog.zthresh < 0
                h = figure; plot(zscore); title('Select Eye Blink Z-Score Threshold');
                [~,cfg.(fields{ii}).eog.zthresh] = ginput(1);
                close(h);
            end
            blink = zscore > cfg.(fields{ii}).eog.zthresh;

            % Ignore zero-padding effects
            blink(1:cfg.(fields{ii}).eog.bpfiltord) = 0;
            blink(end-cfg.(fields{ii}).eog.bpfiltord+1:end) = 0;

            c_art = data.(fields{ii}){jj}(blink,:)'*data.(fields{ii}){jj}(blink,:); % Artifact cov
            c_full = data.(fields{ii}){jj}'*data.(fields{ii}){jj};                  % Full cov
            [todss,pwr0,pwr1]=nt_dss0(c_full,c_art);                                % DSS matrix
            p1=pwr1./pwr0;
            todssinv = pinv(todss)';
            if cfg.(fields{ii}).eog.dssthresh < 0
                h = figure; plot(p1./max(p1),'o'); title('Select Eye Blink Component Threshold');
                button = 3;
                while button ~= 1
                    figure(h);
                    [~,cfg.(fields{ii}).eog.dssthresh,button] = ginput(1);
                    comp_no = find(p1/max(p1)>cfg.(fields{ii}).eog.dssthresh); comp_no = comp_no(end);
                    if button==3 && ...
                            isfield(cfg.(fields{ii}),'layout') && ...
                            exist('ft_defaults','file') && ...
                            length(dim.(fields{ii})) == 2 && any(strcmp(dim.(fields{ii}),'chan')) && ...
                            isfield(data.dim,'chan') && isfield(data.dim.chan,fields{ii}) && ...
                            comp_no >= 1 && comp_no <= length(p1)
                        % Plot dss component
                        datadss = []; datadss.fsample = 1; datadss.time = 0;
                        datadss.dimord = 'chan_time';
                        datadss.label = data.dim.chan.(fields{ii}){jj};
                        datadss.avg = todssinv(:,comp_no);
                        ft_defaults;
                        cfgtmp = []; cfgtmp.parameter = 'avg'; cfg.zlim = 'maxabs';
                        cfgtmp.interactive = 'no'; cfgtmp.layout = cfg.(fields{ii}).layout;
                        cfgtmp.comment = ['DSS component ' num2str(comp_no)]; cfgtmp.verbose = 'off';
                        h_top=figure;
                        try
                            ft_topoplotER(cfgtmp,datadss);
                        catch EX
                            nt_topoplot(cfg.(fields{ii}).layout,todssinv(1:60,comp_no));
                        end
                        uiwait(h_top);
                    end
                end
                close(h)
            end
            artix = find(p1/max(p1)>cfg.(fields{ii}).eog.dssthresh);
            z=nt_mmat(data.(fields{ii}){jj},todss);
            
            if isfield(cfg.(fields{ii}).eog,'realtime')  % Store realtime filtering matrices
                if ~iscell(cfg.(fields{ii}).eog.realtime)
                    cfg.(fields{ii}).eog.realtime = cell(0);
                end
                ix = length(cfg.(fields{ii}).eog.realtime);
                
                znorm = sqrt(sum(z(:,artix).^2,1)./size(z,1));
                zn = z(:,artix)./repmat(znorm,size(z,1),1);
                cref = zn'*zn./size(z,1);
                cxref = data.(fields{ii}){jj}'*zn./size(z,1);
                
                cfg.(fields{ii}).eog.realtime{ix+1} = [];
                cfg.(fields{ii}).eog.realtime{ix+1}.todss = todss(:,artix)./repmat(znorm,size(todss,1),1);
                cfg.(fields{ii}).eog.realtime{ix+1}.r = nt_regcov(cxref,cref);
                
            end
            
            data.(fields{ii}){jj} = nt_tsr(data.(fields{ii}){jj},z(:,artix));
            data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},sz);    % Restore dimensions
        end
    end
    
    % Perform robust PCA for muscle artifacts
    if strcmp(cfg.(fields{ii}).robustpca,'yes')
        fprintf('Performing robust PCA on field: %s\n', fields{ii});
        assert(exist('inexact_alm_rpca','file')==2, ...
            ['Please add inexact_alm_rpca package ' ...
            '(http://perception.csl.illinois.edu/matrix-rank/sample_code.html) to the ' ...
            'Matlab path.']);   % Cannot publicly distribute due to lack of license
        for jj = cells
            % Make data 2D
            sz = size(data.(fields{ii}){jj});
            data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},sz(1),prod(sz(2:end)));
            
            data.(fields{ii}){jj} = inexact_alm_rpca(data.(fields{ii}){jj});
            
            data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},sz);    % Restore dimensions
        end
    end
    
    % Perform STAR for sparse-time artifacts
    if isfield(cfg.(fields{ii}),'star')
        if ~isfield(cfg.(fields{ii}).star,'thresh'); cfg.(fields{ii}).star.thresh = 1; end;
        if ~isfield(cfg.(fields{ii}).star,'radius') || isinf(cfg.(fields{ii}).star.radius)
            cfg.(fields{ii}).star.radius = Inf;
            closest = [];
        else
            assert(exist('ft_defaults','file')==2, ...
                'The FieldTrip toolbox must be in the MATLAB path to use star.radius.');
            assert(isfield(cfg.(fields{ii}).star,'layout'), 'A layout file must be specified for STAR.');
            assert(length(dim.(fields{ii}))==2,'Data can only have time and chan dimensions.');
            ft_defaults;
            closest = nt_proximity(cfg.(fields{ii}).star.layout,cfg.(fields{ii}).star.radius);
            
            % Keep only channels in the layout file
            cfgtmp = [];
            cfgtmp.layout = cfg.(fields{ii}).star.layout;
            layout = ft_prepare_layout(cfgtmp);
            
            cfgtmp = [];
            cfgtmp.(fields{ii}).cell = cells;
            cfgtmp.(fields{ii}).dim = 'chan';
            cfgtmp.(fields{ii}).select = layout.label;
            data = co_selectdim(cfgtmp,data);
        end
        if ~isfield(cfg.(fields{ii}).star,'depth'); cfg.(fields{ii}).star.depth = 1; end;
        
        for jj = cells
            data.(fields{ii}){jj} = nt_star(data.(fields{ii}){jj}, ...
                cfg.(fields{ii}).star.thresh,closest,cfg.(fields{ii}).star.depth);
        end
    end
    
    % Rearrange dimensions to original order
    cfgtmp = [];
    cfgtmp.(fields{ii}).shift = -(timeix-1);
    data = co_shiftdim(cfgtmp,data);
end

% Save cfg settings for future reference
data = co_logcfg(cfg,data);
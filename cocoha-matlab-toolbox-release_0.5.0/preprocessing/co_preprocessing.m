function data = co_preprocessing(cfg,data)
% CO_PREPROCESSING applies user-specified preprocessing steps to time-series data.
%
% INPUTS:
% cfg.dataset           = string with .mat filename. File should contain single variable 
%                         holding data structure.
%
% cfg.FIELD.dataset     = string with filename. If this is a .mat file, and the fields 
%                         below are specified, it will be read directly. 
%                         Otherwise, the FIELDTRIP toolbox will be used to read it. 
% cfg.FIELD.datafield   = name of variable containing time-series data
% cfg.FIELD.eventfield  = name of variable containing event (trigger) data
% cfg.FIELD.eventformat = ['trace'] if events are recorded as a time series (like most MEG 
%                         systems).
% cfg.FIELD.fsfield     = name of variable containing sampling rate, or the sampling rate itself.
% cfg.FIELD.dim         = 'time_chan'/'chan_time'. By default, the longest dimension will be 
%                         assumed to be the time dimension.
% cfg.FIELD.channels    = ['all']/{'all','-A1', ...}, which channels to include/exclude.
%
% cfg.FIELD.demean      = 'yes'/['no'] for demeaning the data.
% cfg.FIELD.detrend     = ['no'] if a scalar is specified, data will be detrended with the specified
%                         polynomial order using NT_DETREND. If a 3-element array is specified,
%                         NT_DETREND_ROBUST will be used to perform robust detrending with
%                         specifications for the threshold for the number of outliers and number of
%                         iterations (default 2 s.d. and 2 iterations).
%
% cfg.FIELD.smooth      = ['no']/scalar smooths data along the time dimension by the specified
%                         number of samples. Useful for performing filtering on line noise
%                         harmonics using kernel of length sample rate / line frequency. Also useful
%                         as a downsampling antialias filter using kernel length of oldfs / newfs.
%
% cfg.FIELD.bpfilter    = 'yes'/['no'] for bandpass filtering of data.FIELD
% cfg.FIELD.hpfilter    = 'yes'/['no'] for highpass filtering of data.FIELD
% cfg.FIELD.lpfilter    = 'yes'/['no'] for lowpass filtering of data.FIELD
% cfg.FIELD.bpfilttype  = 'firws'/'fir1'/'butter'
% cfg.FIELD.hpfilttype  = 'firws'/'fir1'/'butter'
% cfg.FIELD.lpfilttype  = 'firws'/'fir1'/'butter'
% cfg.FIELD.bpfreq      = [low high] bandpass filter frequencies in Hz
% cfg.FIELD.hpfreq      = [high] highpass filter frequency in Hz
% cfg.FIELD.lpfreq      = [low] lowpass filter frequency in Hz
% cfg.FIELD.bpfiltord   = bandpass filter order
% cfg.FIELD.hpfiltord   = highpass filer order
% cfg.FIELD.lpfiltord   = lowpass filter order
% cfg.FIELD.bpfiltwintype = ['blackman'] for firws and ['hamming'] for fir1. Filter window type.
% cfg.FIELD.hpfiltwintype = ['blackman'] for firws and ['hamming'] for fir1. Filter window type.
% cfg.FIELD.lpfiltwintype = ['blackman'] for firws and ['hamming'] for fir1. Filter window type.
% cfg.FIELD.bpfiltdir   = filter direction ['twopass'], 'onepass' or 'onepass-zerophase' (default
%                         for firws).
% cfg.FIELD.hpfiltdir   = filter direction ['twopass'], 'onepass' or 'onepass-zerophase' (default
%                         for firws).
% cfg.FIELD.lpfiltdir   = filter direction ['twopass'], 'onepass' or 'onepass-zerophase' (default
%                         for firws).
% cfg.FIELD.plotfiltresp= 'yes'/['no'] for plotting filter response (firws only)
%
% cfg.FIELD.reref       = 'yes'/['no'] for rereferencing of channel data.FIELD
% cfg.FIELD.refchannel  = 'all' for average rereferencing or an array of
%                         channel #'s, or a cell array of channel names. If only 
%                         one channel is specified, it will be removed.
%
% cfg.FIELD.zscore      = 'yes'/['no'] normalize channels by standard deviation.
%
% cfg.FIELD.hilbert     = ['no']/'abs'/'complex'/'real'/'imag'/'absreal'/'absimag'/
%                         'angle'
%
% data                  = data strucure with time x channel
%
%
% OUTPUTS:
% data                  =
%
%
% See also: CO_SELECTDIM
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU/Oticon, UCL, ETH Zurich
% Author(s): Daniel D.E. Wong

if ~isstruct(cfg); return; end;

if isfield(cfg,'dataset')
    if isempty(data)
        load(cfg.dataset);
        assert(~isempty(data) & isstruct(data), 'Invalid dataset.');    % Simple check
    end
end

% Check data that is already loaded
fields = fieldnames(cfg);
cfgtmp = cfg;
for ii = 1:length(fields)
    if isfield(cfg.(fields{ii}),'dataset') && ~isfield(data,fields{ii})
        cfgtmp = rmfield(cfgtmp,fields{ii});
    end
end

% Init data if empty
if isempty(data)
    data.dim = [];
    data.fsample = [];
    data.event = [];
    data.cfg = {};
end
dim = co_checkdata(cfgtmp,data,'dataset');

for ii = 1:length(fields)
    % Data import - don't overwrite if field already exists in data
    if isfield(cfg.(fields{ii}),'dataset') && ~isfield(data,fields{ii})
        [~,~,ext] = fileparts(cfg.(fields{ii}).dataset);
        
        if strcmp(ext,'.mat') && isfield(cfg.(fields{ii}),'datafield') && isfield(cfg.(fields{ii}),'fsfield')
            % MAT file
            mat = load(cfg.(fields{ii}).dataset);
            
            data.(fields{ii}){1} = mat.(cfg.(fields{ii}).datafield);
            assert(length(size(data.(fields{ii})))==2, 'Imported data should have only 2 dimensions.');
            if isnumeric(cfg.(fields{ii}).fsfield)
                data.fsample.(fields{ii}) = cfg.(fields{ii}).fsfield;
            else
                data.fsample.(fields{ii}) = mat.(cfg.(fields{ii}).fsfield);
            end
            
            % Event data
            if isfield(cfg.(fields{ii}),'eventfield')
                if ~isfield(cfg.(fields{ii}),'eventformat'); cfg.(fields{ii}).eventformat = 'trace'; end;
                if strcmp(cfg.(fields{ii}).eventformat,'trace')
                    tr = mat.(cfg.(fields{ii}).eventfield)(:);
                    event = find(tr);
                    devent = diff(event);
                    onsets = [1; find(devent>1)+1]; % Trigger onsets
                    event = event(onsets);
                    data.event.(fields{ii})(1).sample = event;
                    data.event.(fields{ii})(1).value = strread(num2str(tr(event)'),'%s');   % Need strread to deal with cell structure
                end
            end
        elseif any(strcmp(ext,{'.wav','.ogg','.flac','.au','.mp3','.m4a','.mp4'}))
            if ~exist('audioread.m','file')
                [data.(fields{ii}){1}, data.fsample.(fields{ii})] = wavread(cfg.(fields{ii}).dataset);  % Deprecated MATLAB function
            else
                [data.(fields{ii}){1}, data.fsample.(fields{ii})] = audioread(cfg.(fields{ii}).dataset);
            end
            data.dim.(fields{ii}) = 'time_chan';    % Force dimension
        else
            % FieldTrip Import
            assert(~isempty(which('ft_defaults')),'FieldTrip must be on the MATLAB path to import data.');
            ft_defaults
            
            data.(fields{ii}){1} = ft_read_data(cfg.(fields{ii}).dataset)';
            hdr = ft_read_header(cfg.(fields{ii}).dataset);
            data.fsample.(fields{ii}) = hdr.Fs;
            data.dim.chan.(fields{ii}){1} = hdr.label;
            data.dim.(fields{ii}) = 'time_chan';
            
            try % Read event information if available
                event = ft_read_event(cfg.(fields{ii}).dataset);
                data.event.(fields{ii})(1).sample = zeros(length(event),1);
                data.event.(fields{ii})(1).value = cell(length(event),1);
                for jj = 1:length(event)
                    data.event.(fields{ii})(1).sample(jj) = event(jj).sample;
                    data.event.(fields{ii})(1).value{jj} = event(jj).value;
                end
                clear event;
            end
        end
        
        % Define dimension order from cfg or guess if unable to determine
        if ~isfield(data.dim,fields{ii})
            if ~isfield(cfg.(fields{ii}),'dim')
                [~,ix] = max(size(data.(fields{ii}){1}));
                if ix == 2; data.dim.(fields{ii}) = 'chan_time'; else data.dim.(fields{ii}) = 'time_chan'; end;
            else
                data.dim.(fields{ii}) = cfg.(fields{ii}).dim;
            end
        end

        dim.(fields{ii}) = co_strsplit(data.dim.(fields{ii}),'_');
        
        % Define sensor labels
        if ~isfield(data.dim,'chan') || ~isfield(data.dim.chan,fields{ii})
            chanix = find(strcmp('chan',dim.(fields{ii})));
            chanlabel = num2str([1:size(data.(fields{ii}){1},chanix)]');
            data.dim.chan.(fields{ii}){1} = strtrim(mat2cell(chanlabel, ones(1,size(chanlabel,1)),size(chanlabel,2))); %strread(num2str(1:size(data.(fields{ii}){1},chanix)),'%s');
        end
    end
    
    
    % Set default configuration for filtering
    if ~isfield(cfg.(fields{ii}),'channels'); cfg.(fields{ii}).channels='all'; end;
    if ~isfield(cfg.(fields{ii}),'bpfilter'); cfg.(fields{ii}).bpfilter='no'; end;
    if ~isfield(cfg.(fields{ii}),'hpfilter'); cfg.(fields{ii}).hpfilter='no'; end;
    if ~isfield(cfg.(fields{ii}),'lpfilter'); cfg.(fields{ii}).lpfilter='no'; end;
    if ~isfield(cfg.(fields{ii}),'plotfiltresp'); cfg.(fields{ii}).plotfiltresp='no'; end;
    if ~isfield(cfg.(fields{ii}),'demean'); cfg.(fields{ii}).demean='no'; end;
    if ~isfield(cfg.(fields{ii}),'detrend'); cfg.(fields{ii}).detrend='no'; end;
    if ~isfield(cfg.(fields{ii}), 'smooth'); cfg.(fields{ii}).smooth='no'; end;
    if ~isfield(cfg.(fields{ii}),'reref'); cfg.(fields{ii}).reref='no'; end;
    if ~isfield(cfg.(fields{ii}),'zscore'); cfg.(fields{ii}).zscore='no'; end;
    if ~isfield(cfg.(fields{ii}),'hilbert'); cfg.(fields{ii}).hilbert='no'; end;
    
    
    % Channel selection
    cfgtmp = [];
    cfgtmp.(fields{ii}).cell = 'all';
    cfgtmp.(fields{ii}).dim = 'chan';
    cfgtmp.(fields{ii}).select = cfg.(fields{ii}).channels;
    data = co_selectdim(cfgtmp,data);
    
    % Set time dimension to be first
    timeix = find(strcmp('time',dim.(fields{ii})));
    assert(~isempty(timeix),['dim.' fields{ii} ' is missing the time dimension.']);
    if timeix > 1
        for jj = 1:length(data.(fields{ii})); data.(fields{ii}){jj} = shiftdim(data.(fields{ii}){jj},timeix-1); end;
    end
    
    % Specify trigger channel
    if isfield(cfg.(fields{ii}),'trigchan')
        if ~isfield(cfg.(fields{ii}),'trigthresh'); cfg.(fields{ii}).trigthresh = 0.5; end;
        if ischar(cfg.(fields{ii}).trigchan)
            if isfield(data.dim,'chan') && isfield(data.dim.chan,fields{ii})
                trigix = find(strcmp(cfg.(fields{ii}).trigchan,data.dim.chan));
            else
                error('Cannot specify channel label for cfg.FIELDS.trigchan if data.dim.chan.FIELDS does not exist.');
            end
        elseif isscalar(cfg.(fields{ii}).trigchan)
            trigix = cfg.(fields{ii}).trigchan;
        else
            error('Invalid cfg.FIELDS.trigchan format.');
        end
        
        for jj = 1:length(data.(fields{ii}))
            trig = data.(fields{ii}){jj}(:,trigix) > cfg.(fields{ii}).trigthresh;
            for kk = length(trig)-1:-1:1
                if trig(kk); trig(kk+1) = 0; end;
                trig = find(trig);
            end
        end
    end
    
    % Store sizes and reshape to 2D
    sz = cell(1,length(data.(fields{ii})));
    for jj = 1:length(sz)
        sz{jj} = size(data.(fields{ii}){jj});
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},sz{jj}(1),prod(sz{jj}(2:end)));
    end
    
    % Demean
    if strcmp(cfg.(fields{ii}).demean,'yes')
        for jj = 1:length(data.(fields{ii}))
            data.(fields{ii}){jj} = data.(fields{ii}){jj} - ...
                repmat(mean(data.(fields{ii}){jj},1), size(data.(fields{ii}){jj},1),1);
        end
    end
    
    % Detrend
    if isnumeric(cfg.(fields{ii}).detrend)
        for jj = 1:length(data.(fields{ii}))
            if ~exist('nt_detrend','var')
                addpath(fullfile(fileparts(which('co_defaults')), 'external', 'NoiseTools'));
                addpath(fullfile(fileparts(which('co_defaults')), 'external', 'NoiseTools', 'COMPAT'));
            end
            if length(cfg.(fields{ii}).detrend) == 1
                data.(fields{ii}){jj} = nt_detrend(data.(fields{ii}){jj},cfg.(fields{ii}).detrend);
            else
                assert(length(cfg.(fields{ii}).detrend)==3, ...
                    'A 2-element array must be specified for robust detrending.');
                data.(fields{ii}){jj} = nt_detrend_robust(data.(fields{ii}){jj}, ...
                    cfg.(fields{ii}).detrend(1), [], [], cfg.(fields{ii}).detrend(2), ...
                    cfg.(fields{ii}).detrend(3));
            end
        end
    end
    
    % Smooth
    if isnumeric(cfg.(fields{ii}).smooth)
        if ~exist('nt_smooth','var')
            addpath(fullfile(fileparts(which('co_defaults')), 'external', 'NoiseTools'));
            addpath(fullfile(fileparts(which('co_defaults')), 'external', 'NoiseTools', 'COMPAT'));
        end
        for jj = 1:length(data.(fields{ii}))
            data.(fields{ii}){jj} = nt_smooth(data.(fields{ii}){jj},cfg.(fields{ii}).smooth);
        end
    end
    
    % Bandpass filtering
    if strcmp(cfg.(fields{ii}).bpfilter,'yes')
        assert(length(cfg.(fields{ii}).bpfreq)==2, ['cfg.' fields{ii} '.bpfreq should be specified as [low high] in Hz.']);
        
        % Filter function
        switch cfg.(fields{ii}).bpfilttype
            case 'firws'
                if ~isfield(cfg.(fields{ii}),'bpfiltdir'); cfg.(fields{ii}).bpfiltdir = 'onepass-zerophase'; end;
%                 assert(strcmp(cfg.(fields{ii}).bpfiltdir,'onepass-zerophase'), 'Unsupported filter direction specification for firws');
                filtfun = 'firfilt';
            case {'fir1','butter'}
                if ~isfield(cfg.(fields{ii}),'bpfiltdir'); cfg.(fields{ii}).bpfiltdir = 'twopass'; end;
                if strcmp(cfg.(fields{ii}).bpfiltdir,'twopass')
                    filtfun = 'filtfilt';
                elseif strcmp(cfg.(fields{ii}).bpfiltdir,'onepass')
                    filtfun = 'filter';
                elseif strcmp(cfg.(fields{ii}).bpfilter,'onepass-zerophase') && ~strcmp(cfg.(fields{ii}).bpfilttype,'butter')
                    filtfun = 'firfilt';
                else
                    error(['Unsupported filter direction specification for ', cfg.(fields{ii}).bpfilttype]);
                end
            otherwise
                error(['cfg.' fields{ii} '.bpfilttype not supported'])
        end
%         if isfield(cfg.(fields{ii}),'bpfiltdir'); filtfun = cfg.(fields{ii}).bpfiltdir; end;
        
        % Window for filtering
        window = {};
        if isfield(cfg.(fields{ii}),'bpfiltwintype'); window = {feval(cfg.(fields{ii}).bpfiltwintype,cfg.(fields{ii}).bpfiltord+1)}; end;
        if strcmp(cfg.(fields{ii}).bpfilttype,'firws')
            if ~exist('firws.m','file'); addpath(fullfile(fileparts(which('co_defaults')), 'external', 'firfilt')); end;
            cfg.(fields{ii}).b = firws(cfg.(fields{ii}).bpfiltord,2*cfg.(fields{ii}).bpfreq/data.fsample.(fields{ii}),window{:});
            for jj = 1:length(data.(fields{ii})); data.(fields{ii}){jj} = feval(filtfun,cfg.(fields{ii}).b,1,data.(fields{ii}){jj}); end;
            if strcmp(cfg.(fields{ii}).plotfiltresp,'yes'); figure, plotfresp(cfg.(fields{ii}).b,1,[],data.fsample.(fields{ii})); end;
        elseif strcmp(cfg.(fields{ii}).bpfilttype,'fir1')
            cfg.(fields{ii}).b = fir1(cfg.(fields{ii}).bpfiltord,2*cfg.(fields{ii}).bpfreq/data.fsample.(fields{ii}),'bandpass',window{:});
            for jj = 1:length(data.(fields{ii})); data.(fields{ii}){jj} = feval(filtfun,cfg.(fields{ii}).b,1,data.(fields{ii}){jj}); end;
            if strcmp(cfg.(fields{ii}).plotfiltresp,'yes'); fvtool(cfg.(fields{ii}).b,1); end;
        elseif strcmp(cfg.(fields{ii}).bpfilttype,'butter')
            [cfg.(fields{ii}).b,cfg.(fields{ii}).a] = butter(cfg.(fields{ii}).bpfiltord,2*cfg.(fields{ii}).bpfreq/data.fsample.(fields{ii}),'bandpass');
            assert(isempty(window),'Butterworth filter does not use a window.');
            for jj = 1:length(data.(fields{ii})); data.(fields{ii}){jj} = feval(filtfun,cfg.(fields{ii}).b,cfg.(fields{ii}).a,data.(fields{ii}){jj}); end;
            if strcmp(cfg.(fields{ii}).plotfiltresp,'yes'); fvtool(cfg.(fields{ii}).b,cfg.(fields{ii}).a); end;
        end
    end

    % High pass filtering
    if strcmp(cfg.(fields{ii}).hpfilter,'yes')
        assert(length(cfg.(fields{ii}).hpfreq)==1, 'cfg.hpfreq should be specified as [high] in Hz.');
        
        % Filter function
        switch cfg.(fields{ii}).hpfilttype
            case 'firws'
                if ~isfield(cfg.(fields{ii}),'hpfiltdir'); cfg.(fields{ii}).hpfiltdir = 'onepass-zerophase'; end;
%                 assert(strcmp(cfg.(fields{ii}).hpfiltdir,'onepass-zerophase'), 'Unsupported filter direction specification for firws');
                filtfun = 'firfilt';
            case {'fir1','butter'}
                if ~isfield(cfg.(fields{ii}),'hpfiltdir'); cfg.(fields{ii}).hpfiltdir = 'twopass'; end;
                if strcmp(cfg.(fields{ii}).hpfiltdir,'twopass')
                    filtfun = 'filtfilt';
                elseif strcmp(cfg.(fields{ii}).hpfiltdir,'onepass')
                    filtfun = 'filter';
                elseif strcmp(cfg.(fields{ii}).hpfilter,'onepass-zerophase') && ~strcmp(cfg.(fields{ii}).hpfilttype,'butter')
                    filtfun = 'firfilt';
                else
                    error(['Unsupported filter direction specification for ', cfg.(fields{ii}).hpfilttype]);
                end
            otherwise
                error(['cfg.' fields{ii} '.hpfilttype not supported'])
        end
%         if isfield(cfg.(fields{ii}),'hpfiltdir'); filtfun = cfg.(fields{ii}).hpfiltdir; end;
        
        % Window for filtering
        window = {};
        if isfield(cfg.(fields{ii}),'hpfiltwintype'); window = {feval(cfg.(fields{ii}).hpfiltwintype,cfg.(fields{ii}).hpfiltord+1)}; end;
        if strcmp(cfg.(fields{ii}).hpfilttype,'firws')
            if ~exist('firws.m','file'); addpath(fullfile(fileparts(which('co_defaults')), 'external', 'firfilt')); end;
            cfg.(fields{ii}).b = firws(cfg.(fields{ii}).hpfiltord,2*cfg.(fields{ii}).hpfreq/data.fsample.(fields{ii}),'high');
            for jj = 1:length(data.(fields{ii})); data.(fields{ii}){jj} = feval(filtfun,cfg.(fields{ii}).b,1,data.(fields{ii}){jj}); end;
            if strcmp(cfg.(fields{ii}).plotfiltresp,'yes'); figure, plotfresp(cfg.(fields{ii}).b,1,[],data.fsample.(fields{ii})); end;
        elseif strcmp(cfg.(fields{ii}).hpfilttype,'fir1')
            cfg.(fields{ii}).b = fir1(cfg.(fields{ii}).hpfiltord,2*cfg.(fields{ii}).hpfreq/data.fsample.(fields{ii}),'high',window{:});
            for jj = 1:length(data.(fields{ii})); data.(fields{ii}){jj} = feval(filtfun,cfg.(fields{ii}).b,1,data.(fields{ii}){jj}); end;
            if strcmp(cfg.(fields{ii}).plotfiltresp,'yes'); fvtool(cfg.(fields{ii}).b,1); end;
        elseif strcmp(cfg.(fields{ii}).hpfilttype,'butter')
            [cfg.(fields{ii}).b,cfg.(fields{ii}).a] = butter(cfg.(fields{ii}).hpfiltord,2*cfg.(fields{ii}).hpfreq/data.fsample.(fields{ii}),'high');
            assert(isempty(window),'Butterworth filter does not use a window.');
            for jj = 1:length(data.(fields{ii})); data.(fields{ii}){jj} = feval(filtfun,cfg.(fields{ii}).b,cfg.(fields{ii}).a,data.(fields{ii}){jj}); end;
            if strcmp(cfg.(fields{ii}).plotfiltresp,'yes'); fvtool(cfg.(fields{ii}).b,cfg.(fields{ii}).a); end;
        else
            error(['cfg.' fields{ii} '.hpfilttype not supported'])
        end
    end

    % Low-pass filtering
    if strcmp(cfg.(fields{ii}).lpfilter,'yes')
        assert(length(cfg.(fields{ii}).lpfreq)==1, 'cfg.hpfreq should be specified as [high] in Hz.');
        
        % Filter function
        switch cfg.(fields{ii}).lpfilttype
            case 'firws'
                if ~isfield(cfg.(fields{ii}),'lpfiltdir'); cfg.(fields{ii}).lpfiltdir = 'onepass-zerophase'; end;
%                 assert(strcmp(cfg.(fields{ii}).lpfiltdir,'onepass-zerophase'), 'Unsupported filter direction specification for firws');
                filtfun = 'firfilt';
            case {'fir1','butter'}
                if ~isfield(cfg.(fields{ii}),'lpfiltdir'); cfg.(fields{ii}).lpfiltdir = 'twopass'; end;
                if strcmp(cfg.(fields{ii}).lpfiltdir,'twopass')
                    filtfun = 'filtfilt';
                elseif strcmp(cfg.(fields{ii}).lpfiltdir,'onepass')
                    filtfun = 'filter';
                elseif strcmp(cfg.(fields{ii}).lpfilter,'onepass-zerophase') && ~strcmp(cfg.(fields{ii}).lpfilttype,'butter')
                    filtfun = 'firfilt';
                else
                    error(['Unsupported filter direction specification for ', cfg.(fields{ii}).lpfilttype]);
                end
            otherwise
                error(['cfg.' fields{ii} '.lpfilttype not supported'])
        end
%         if isfield(cfg.(fields{ii}),'lpfiltdir'); filtfun = cfg.(fields{ii}).lpfiltdir; end;
        
        % Window for filtering
        window = {};
        if isfield(cfg.(fields{ii}),'lpfiltwintype'); window = {feval(cfg.(fields{ii}).lpfiltwintype,cfg.(fields{ii}).lpfiltord+1)}; end;
        if strcmp(cfg.(fields{ii}).lpfilttype,'firws')
            if ~exist('firws.m','file'); addpath(fullfile(fileparts(which('co_defaults')), 'external', 'firfilt')); end;
            cfg.(fields{ii}).b = firws(cfg.(fields{ii}).lpfiltord,2*cfg.(fields{ii}).lpfreq/data.fsample.(fields{ii}));
            for jj = 1:length(data.(fields{ii})); data.(fields{ii}){jj} = feval(filtfun,cfg.(fields{ii}).b,1,data.(fields{ii}){jj}); end;
            if strcmp(cfg.(fields{ii}).plotfiltresp,'yes'); figure, plotfresp(cfg.(fields{ii}).b,1,[],data.fsample.(fields{ii})); end;
        elseif strcmp(cfg.(fields{ii}).lpfilttype,'fir1')
            cfg.(fields{ii}).b = fir1(cfg.(fields{ii}).lpfiltord,2*cfg.(fields{ii}).lpfreq/data.fsample.(fields{ii}),'low',window{:});
            for jj = 1:length(data.(fields{ii})); data.(fields{ii}){jj} = feval(filtfun,cfg.(fields{ii}).b,1,data.(fields{ii}){jj}); end;
            if strcmp(cfg.(fields{ii}).plotfiltresp,'yes'); fvtool(cfg.(fields{ii}).b,1); end;
        elseif strcmp(cfg.(fields{ii}).lpfilttype,'butter')
            [cfg.(fields{ii}).b,cfg.(fields{ii}).a] = butter(cfg.(fields{ii}).lpfiltord,2*cfg.(fields{ii}).lpfreq/data.fsample.(fields{ii}),'low');
            assert(isempty(window),'Butterworth filter does not use a window.');
            for jj = 1:length(data.(fields{ii})); data.(fields{ii}){jj} = feval(filtfun,cfg.(fields{ii}).b,cfg.(fields{ii}).a,data.(fields{ii}){jj}); end;
            if strcmp(cfg.(fields{ii}).plotfiltresp,'yes'); fvtool(cfg.(fields{ii}).b,cfg.(fields{ii}).a); end;
        else
            error(['cfg.' fields{ii} '.lpfilttype not supported'])
        end
    end
    
    % Rereference
    if strcmp(cfg.(fields{ii}).reref,'yes')
        assert(isfield(cfg.(fields{ii}),'refchannel'), ['cfg.' fields{ii} '.refchannel must be specified']);
        chanix = find(strncmp('chan',dim.(fields{ii}),4));
        assert(~isempty(chanix),['No channel dimension in data.dim.' fields{ii}]);
        chanix = mod(chanix-timeix,length(dim.(fields{ii})))+1;   % Determine new position wrt time
        
        if ~isnumeric(cfg.(fields{ii}).refchannel) && ~iscell(cfg.(fields{ii}).refchannel)
            cfg.(fields{ii}).refchannel = {cfg.(fields{ii}).refchannel};    % Might be a single channel specified as a string?
        end
        
        for jj = 1:length(data.(fields{ii}))
            % Reshape data to N-D, shift chanix to first dim, then reshape to 2-D
            data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},sz{jj});
            data.(fields{ii}){jj} = shiftdim(data.(fields{ii}){jj},chanix-1);
            rerefsz = size(data.(fields{ii}){jj});
            data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},rerefsz(1),prod(rerefsz(2:end)));
            
            % Determine reference channel indexes from list of labels
            if isnumeric(cfg.(fields{ii}).refchannel)   % Array list of channel indexes
                refix = intersect(cfg.(fields{ii}).refchannel,1:size(data.(fields{ii}){jj},1));
            else                                        % Cell list of channel labels / 'all'
                refix = [];
                for kk = 1:length(cfg.(fields{ii}).refchannel)
                    if strcmp(cfg.(fields{ii}).refchannel{kk},'all')
                        refix = 1:length(data.dim.chan.(fields{ii}){jj});
                    else
                        assert(isfield(data.dim,'chan') & isfield(data.dim.chan,fields{ii}), 'Missing channel labels in data.dim.chan.');
                        ix = find(strcmp(cfg.(fields{ii}).refchannel{jj},data.dim.chan.(fields{ii}){jj}));
                        refix = [refix ix];
                    end
                end
                refix = unique(refix);
            end

            ref = mean(data.(fields{ii}){jj}(refix,:),1);
            data.(fields{ii}){jj} = data.(fields{ii}){jj} - repmat(ref,size(data.(fields{ii}){jj},1),1);

            % If a single reference channel was specified, remove it
            if length(refix) == 1
                data.(fields{ii}){jj}(refix,:) = [];
                if jj==1; data.dim.chan.(fields{ii}){jj}(refix) = []; sz{jj}(chanix)=sz{jj}(chanix)-1; rerefsz(1) = rerefsz(1)-1; end;
            end
            
            % Reshape data to N-D, then shift so time is first dim again, then reshape to 2-D
            data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},rerefsz);
            data.(fields{ii}){jj} = shiftdim(data.(fields{ii}){jj},length(sz{jj})-chanix+1);
            data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},sz{jj}(1),prod(sz{jj}(2:end)));
        end
    end
    
    % Normalize by standard dev
    if strcmp(cfg.(fields{ii}).zscore,'yes')
        for jj = 1:length(data.(fields{ii}))
            data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},sz{jj}(1),prod(sz{jj}(2:end)));
            data.(fields{ii}){jj} = (data.(fields{ii}){jj}-repmat(mean(data.(fields{ii}){jj},1),sz{jj}(1),1))./ ...
                repmat(std(data.(fields{ii}){jj},[],1),sz{jj}(1),1);
            data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},sz{jj});
        end
    end
    
    % Hilbert transform
    if ~strcmp(cfg.(fields{ii}).hilbert,'no')
        for jj = 1:length(data.(fields{ii}))
            data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj}, sz{jj}(1), prod(sz{jj}(2:end)));
            h = hilbert(data.(fields{ii}){jj});
            
            if strcmp(cfg.(fields{ii}).hilbert,'abs')
                data.(fields{ii}){jj} = abs(h);
            elseif strcmp(cfg.(fields{ii}).hilbert,'complex')
                data.(fields{ii}){jj} = h;
            elseif strcmp(cfg.(fields{ii}).hilbert,'real')
                data.(fields{ii}){jj} = real(h);
            elseif strcmp(cfg.(fields{ii}).hilbert,'imag')
                data.(fields{ii}){jj} = imag(h);
            elseif strcmp(cfg.(fields{ii}).hilbert,'absreal')
                data.(fields{ii}){jj} = abs(real(h));
            elseif strcmp(cfg.(fields{ii}).hilbert,'absimag')
                data.(fields{ii}){jj} = abs(imag(h));
            elseif strcmp(cfg.(fields{ii}).hilbert,'angle')
                data.(fields{ii}){jj} = atan(real(h)./imag(h));
            end
        end
    end
    
    % Restore dimensions and rearrange to original order
    for jj = 1:length(data.(fields{ii}))
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},sz{jj});
        if timeix > 1
            data.(fields{ii}){jj} = shiftdim(data.(fields{ii}){jj}, length(dim.(fields{ii}))-timeix+1);
        end
    end
end

% Save cfg settings for future reference
data = co_logcfg(cfg,data);
% [stck,stckix] = dbstack;
% cfg.fcn = stck(stckix).name;
% cfg.date = date;
% if ~isfield(data,'cfg'); data.cfg{1} = cfg; else data.cfg{end+1} = cfg; end;


function data = firfilt(b,a,data,nframes)   % Adaptation of firfilt.m from Andreas Widmann's firfilt toolbox
if ~exist('firws.m','file'); addpath(fullfile(fileparts(which('co_defaults')), 'external', 'firfilt')); end;
assert(length(a) == 1 && a == 1,'firfilt can only be used for FIR filters.');

if ~exist('nframes','var'); nframes = 1000; end;

% Filter's group delay
if mod(length(b), 2) ~= 1
    error('Filter order is not even.');
end
groupDelay = (length(b) - 1) / 2;
%dcArray = [1 size(data.eeg,1)+1];

% Pad beginning of data and get initial conditions
ziDataDur = min(groupDelay, size(data,1));
[~,zi] = filter(b, 1, double([data(ones(1,groupDelay),:); ...
    data(1:ziDataDur,:)]'), [], 2);

blockArray = [(groupDelay+1):nframes:size(data,1) size(data,1)+1];

for iBlock = 1:(length(blockArray)-1)   % Filter data
    [temp,zi] = filter(b, 1, double(data(blockArray(iBlock):(blockArray(iBlock+1)-1),:)'), zi, 2);
    data((blockArray(iBlock)-groupDelay):(blockArray(iBlock+1)-groupDelay-1),:) = temp';
end

% Pad end of data with DC constant
temp = filter(b, 1, double(data(ones(1,groupDelay)*size(data,1),:)'), zi, 2);
data((size(data,1)+1-ziDataDur):size(data,1),:) = ...
    temp(:,(end-ziDataDur+1):end)';
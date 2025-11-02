function data = co_freqband(cfg,data)
% CO_FREQBAND splits data into frequency bands. Frequency bands will be added as a new dimension.
% Currently only operates in the time domain.
%
% INPUTS:
% cfg.FIELD.dim     = ['time'] dimension on which to operate. (TODO)
% cfg.FIELD.fdim    = ['freq'] name of newly created frequency dimension.
% cfg.FIELD.filttype= 'firws'/'fir1'/'butter'.
% cfg.FIELD.filtord = filter order. If specified as a scalar, the same order will be applied for all
%                     bands. If specified as length-N vector, different orders can be applied for
%                     each of N frequency bands.
% cfg.FIELD.filtdir = ['twopass'], 'onepass', 'onepass-zerophase' (default for firws). See
%                     CO_PREPROCESSING for more details.
% cfg.FIELD.bands   = Nx2 matrix of N frequency bands. If a band has 0 as the starting frequency, a
%                     low-pass filter will be performed. If a band has Inf as the end frequency, a
%                     high-pass filter will be performed.
%
% data
%
% OUTPUTS:
% data
%
%
% See also: CO_PREPROCESSING
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZH
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);

for ii = 1:length(fields)
    if ~isfield(cfg.(fields{ii}),'dim'); cfg.(fields{ii}).dim = 'time'; end;
    if ~isfield(cfg.(fields{ii}),'fdim'); cfg.(fields{ii}).fdim = 'freq'; end;
    if length(cfg.(fields{ii}).filtord) == 1    % If filtord is a scalar
        cfg.(fields{ii}).filtord = repmat(cfg.(fields{ii}).filtord, ...
            size(cfg.(fields{ii}).bands,1),1);
    end
    
    assert(strcmp(cfg.(fields{ii}).dim,'time'),'Only operations on the time dimension are currently supported.');
    
    % Make specified cfg.FIELDS.dim first
    dimix = find(strcmp(cfg.(fields{ii}).dim,dim.(fields{ii})));
    cfgtmp = [];
    cfgtmp.(fields{ii}).shift = dimix-1;
    data = co_shiftdim(cfgtmp,data);
    
    datafreq = cell(1,length(data.(fields{ii})));
    for jj = 1:length(data.(fields{ii}))
        % Dim order makes filling frequency bands easier - but need to shift later
        datafreq{jj} = zeros([size(cfg.(fields{ii}).bands,1) size(data.(fields{ii}){jj})]);
    end
    for jj = 1:size(cfg.(fields{ii}).bands,1)
        cfgtmp = [];
        if cfg.(fields{ii}).bands(jj,1) == 0
            % Lowpass
            cfgtmp.(fields{ii}).lpfilter = 'yes';
            cfgtmp.(fields{ii}).lpfilttype = cfg.(fields{ii}).filttype;
            cfgtmp.(fields{ii}).lpfiltord = cfg.(fields{ii}).filtord(jj);
            cfgtmp.(fields{ii}).lpfreq = cfg.(fields{ii}).bands(jj,2);
            if isfield(cfg.(fields{ii}),'filtdir')
                cfgtmp.(fields{ii}).lpfiltdir = cfg.(fields{ii}).filtdir;
            end
        elseif cfg.(fields{ii}).bands(jj,2) == Inf
            % Highpass
            cfgtmp.(fields{ii}).hpfilter = 'yes';
            cfgtmp.(fields{ii}).hpfilttype = cfg.(fields{ii}).filttype;
            cfgtmp.(fields{ii}).hpfiltord = cfg.(fields{ii}).filtord(jj);
            cfgtmp.(fields{ii}).hpfreq = cfg.(fields{ii}).bands(jj,1);
            if isfield(cfg.(fields{ii}),'filtdir')
                cfgtmp.(fields{ii}).hpfiltdir = cfg.(fields{ii}).filtdir;
            end
        else
            % Bandpass
            cfgtmp.(fields{ii}).bpfilter = 'yes';
            cfgtmp.(fields{ii}).bpfilttype = cfg.(fields{ii}).filttype;
            cfgtmp.(fields{ii}).bpfiltord = cfg.(fields{ii}).filtord(jj);
            cfgtmp.(fields{ii}).bpfreq = cfg.(fields{ii}).bands(jj,:);
            if isfield(cfg.(fields{ii}),'filtdir')
                cfgtmp.(fields{ii}).bpfiltdir = cfg.(fields{ii}).filtdir;
            end
        end
        datatmp = co_preprocessing(cfgtmp,data);
        
        for kk = 1:length(data.(fields{ii}))
            datafreq{kk}(jj,:) = datatmp.(fields{ii}){kk}(:);
            if jj == size(cfg.(fields{ii}).bands,1) % All bands filled - shift dims to proper order
                datafreq{kk} = shiftdim(datafreq{kk},1);
            end
        end
    end
    
    for jj = 1:length(data.(fields{ii}))
        cfgtmp = []; cfgtmp.(fields{ii}).cell = jj; sz = co_size(cfgtmp,datatmp);
        % sz = size(datatmp.(fields{ii}){jj});
        datafreq{jj} = reshape(datafreq{jj},[sz, size(cfg.(fields{ii}).bands,1)]);	% Make sure the shape is right
    end
    data.(fields{ii}) = datafreq;
    data.dim.(fields{ii}) = [data.dim.(fields{ii}) '_' cfg.(fields{ii}).fdim];
    for jj = 1:length(data.(fields{ii}))
        data.dim.(cfg.(fields{ii}).fdim).(fields{ii}){jj} = ...
            mat2cell(mean(cfg.(fields{ii}).bands,2)',1, ones(1,size(cfg.(fields{ii}).bands,1)));
    end
    
    % Shift index order back
    cfgtmp = [];
    cfgtmp.(fields{ii}).shift = -dimix+1;
    data = co_shiftdim(cfgtmp,data);
end

data = co_logcfg(cfg,data);
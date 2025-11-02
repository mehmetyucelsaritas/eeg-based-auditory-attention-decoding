function data = co_lagdata(cfg,data)
% CO_LAGDATA lags a specified dimension. The lags will form a new dimension.
%
% INPUTS:
% cfg.FIELD.dim     = ['time'] the dimension to be lagged.
% cfg.FIELD.lags    = array of sample lags (i.e. -1:1). Negative lags shift the data earlier
%                     (towards the first index, non-causal). Positive lags shift the data later
%                     (causal).
% cfg.FIELD.window  = ['rectangular']/'hann'/'hamming'/'blackman' window to apply to the lags.
% cfg.FIELD.trim    = ['no']/'yes' whether to trim leading/trailing zero padding.
%
% OUTPUTS:
% data
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZ
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);

if ~exist('LagGenerator','file')
    disp('Adding the Telluride Decoding Toolbox to the MATLAB path.');
    addpath(fullfile(fileparts(which('co_defaults')), 'external', 'telluride-decoding-toolbox'));
end

for ii = 1:length(fields)
    if ~isfield(cfg.(fields{ii}),'dim'); cfg.(fields{ii}).dim = 'time'; end;
    assert(isfield(cfg.(fields{ii}),'lags'), 'Missing lag parameters.');
    cfg.(fields{ii}).lags = sort(cfg.(fields{ii}).lags);
    if ~isfield(cfg.(fields{ii}),'window'); cfg.(fields{ii}).window = 'rectangular'; end;
    if ~isfield(cfg.(fields{ii}),'trim'); cfg.(fields{ii}).trim = 'no'; end;
    
    dimix = find(strcmp(cfg.(fields{ii}).dim,dim.(fields{ii})));
    if dimix > 1
        cfgtmp = [];
        cfgtmp.(fields{ii}).shift = dimix-1;
        data = co_shiftdim(cfgtmp,data);
    end
    
    % Insert lags and add as dimension labels
    lagdimname = [cfg.(fields{ii}).dim, 'lag'];
    for jj = 1:length(data.(fields{ii}))
        dimsz = size(data.(fields{ii}){jj});
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},dimsz(1),prod(dimsz(2:end)));
        data.(fields{ii}){jj} = LagGenerator(data.(fields{ii}){jj}, cfg.(fields{ii}).lags);
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},[dimsz, length(cfg.(fields{ii}).lags)]);
        data.dim.(lagdimname).(fields{ii}){jj} = mat2cell(cfg.(fields{ii}).lags(:)', ...
            1,ones(1,length(cfg.(fields{ii}).lags)));
        
        % Apply window
        switch cfg.(fields{ii}).window
            case 'rectangular'
                % Do nothing
            otherwise
                % Half-window
%                 winlags = max(abs(cfg.(fields{ii}).lags)); winlags = -winlags:winlags;
%                 win = feval(cfg.(fields{ii}).window, length(winlags));
%                 [~,winix] = intersect(winlags,cfg.(fields{ii}).lags);
                
                % Full- window
                winlags = min(cfg.(fields{ii}).lags):max(cfg.(fields{ii}).lags);
                win = feval(cfg.(fields{ii}).window, length(winlags));
                [~,winix] = intersect(winlags,cfg.(fields{ii}).lags);
                
                % Temporarily shift lag dim to front to apply window
                data.(fields{ii}){jj} = shiftdim(data.(fields{ii}){jj}, ...
                    ndims(data.(fields{ii}){jj})-1);
                for kk = 1:length(winix)
                    data.(fields{ii}){jj}(kk,:) = win(winix(kk)) * data.(fields{ii}){jj}(kk,:);
                end
                data.(fields{ii}){jj} = shiftdim(data.(fields{ii}){jj},1);
        end
    end
    
    data.dim.(fields{ii}) = [data.dim.(fields{ii}) '_' lagdimname];
    
    if dimix > 1
        cfgtmp = [];
        cfgtmp.(fields{ii}).shift = 1-dimix;
        data = co_shiftdim(cfgtmp,data);
    end
    
    % Trim leading/trailing zero-padding
    if strcmp(cfg.(fields{ii}).trim,'yes')
        for jj = 1:length(data.(fields{ii}))
            cfgtmp = [];
            cfgtmp.(fields{ii}).cell = jj;
            cfgtmp.(fields{ii}).dim = cfg.(fields{ii}).dim;
            dimsz = size(data.(fields{ii}){jj});
            cfgtmp.(fields{ii}).select = 1:dimsz(dimix);
            
            if cfg.(fields{ii}).lags(1) < 0     % Trim trailing
                cfgtmp.(fields{ii}).select = cfgtmp.(fields{ii}).select( ...
                    1:end+cfg.(fields{ii}).lags(1));
            end
            if cfg.(fields{ii}).lags(end) > 0   % Trim leading
                cfgtmp.(fields{ii}).select = cfgtmp.(fields{ii}).select( ...
                    cfg.(fields{ii}).lags(end)+1:end);
                data.event.eeg(jj).sample = data.event.eeg(jj).sample + cfg.(fields{ii}).lags(end);
            end
            data = co_selectdim(cfgtmp,data);
        end
    end
end

% Save cfg settings for future reference
data = co_logcfg(cfg,data);
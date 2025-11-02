function data = co_resampledata(cfg,data)
% CO_RESAMPLEDATA resamples specified data field. If the time dimension is resampled, events will be
% resampled as well. If there are labels associated with the resampled dimension, an attempt will be
% made to resample them as well if they are numeric. Otherwise, the labels will simply be removed.
%
% INPUTS
% cfg.FIELD.dimname     = ['time'], dimension to be resampled
% cfg.FIELD.newfs       = new sampling rate for data.FIELD
% cfg.FIELD.oldfs       = old sampling rate for data.FIELD. Not necessary if dimname is 'time'
% cfg.FIELD.method      = ['fft']/'interp' use FFT resampling or cubic interpolation. Note that 
%                         lowpass filtering at the Nyquist frequency is required beforehand if
%                         'interp' is used.
% data                  =
%
% OUTPUTS
% data                  = resampled data
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZH
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data);
if ~isstruct(cfg); return; end;
fields = fieldnames(cfg);
for ii = 1:length(fields)
    if ~isfield(cfg.(fields{ii}),'dimname'); cfg.(fields{ii}).dimname = 'time'; end;
    if ~isfield(cfg.(fields{ii}),'method'); cfg.(fields{ii}).method = 'fft'; end;
    assert(strcmp(cfg.(fields{ii}).dimname,'time') | isfield(cfg.(fields{ii}),'oldfs'), ...
        ['cfg.' fields{ii} '.oldfs must be specified for non-time dimension resampling.']);
    
    % Rearrange dimensions so operating dimension is 1st
    dimname = cfg.(fields{ii}).dimname;
    dimix = find(strncmp(dimname,dim.(fields{ii}),length(dimname)));
    cfgtmp = []; cfgtmp.(fields{ii}).shift = dimix-1; data = co_shiftdim(cfgtmp,data);
    
    newfs = cfg.(fields{ii}).newfs;
    if isfield(cfg.(fields{ii}),'oldfs')
        oldfs = cfg.(fields{ii}).oldfs;
    else
        oldfs = data.fsample.(fields{ii});
    end
    for jj = 1:length(data.(fields{ii}))
        sz_old = size(data.(fields{ii}){jj});
        t_old = 0:1/oldfs:sz_old(1)/oldfs-1/oldfs;
        t_new = 0:1/newfs:sz_old(1)/oldfs-1/oldfs;
        
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},sz_old(1),prod(sz_old(2:end)));
        
        switch cfg.(fields{ii}).method
            case {'interp'}
                for kk = 1:size(data.(fields{ii}){jj},2)
                    data.(fields{ii}){jj}=interp1(t_old, data.(fields{ii}){jj}(:,kk), t_new, 'pchip', 'extrap');
                end
            case {'fft'}
                data.(fields{ii}){jj}=fft_resample(data.(fields{ii}){jj}, oldfs, newfs);
            otherwise
                error(['cfg.' fields{ii} '.method not supported']);
        end
        
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},[numel(data.(fields{ii}){jj})/prod(sz_old(2:end)) sz_old(2:end)]);
        
        % Adjust event samples if time dimension was altered
        if strcmp(cfg.(fields{ii}).dimname,'time') && isfield(data.event,fields{ii})
            data.event.(fields{ii})(jj).sample = round(data.event.(fields{ii})(jj).sample*newfs/oldfs);
            
            % Handle event resampling of out-of-bound indexes
            data.event.(fields{ii})(jj).sample(data.event.(fields{ii})(jj).sample<1) = 1;
            data.event.(fields{ii})(jj).sample(data.event.(fields{ii})(jj).sample>size(data.(fields{ii}){jj},1)) =  ...
                size(data.(fields{ii}){jj},1);
        end
        
        % Handle dim labels
        if isfield(data.dim,cfg.(fields{ii}).dimname) && isfield(data.dim.(cfg.(fields{ii}).dimname),fields{ii})
            nLabels = cell2mat(data.dim.(cfg.(fields{ii}).dimname).(fields{ii}){jj});
            if isnumeric(nLabels)
                data.dim.(cfg.(fields{ii}).dimname).(fields{ii}){jj} = ...
                    num2cell(interp1(t_old, nLabels, t_new, 'linear', 'extrap'));
            else
                warning(['Unable to resample non-numeric dimension labels for data.' fields{ii} ...
                    '{' jj '}. Labels will simply be removed.']);
            end
        end
    end
    
    % Shift dimensions back to original order
    cfgtmp = []; cfgtmp.(fields{ii}).shift = 1-dimix; data = co_shiftdim(cfgtmp,data);
    
    % Adjust fsample if time dimension was altered
    if strcmp(cfg.(fields{ii}).dimname,'time')
        data.fsample.(fields{ii}) = newfs;
    end
end

% Save cfg settings for future reference
data = co_logcfg(cfg,data);

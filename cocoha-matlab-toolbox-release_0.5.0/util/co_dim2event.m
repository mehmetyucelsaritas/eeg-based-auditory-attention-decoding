function data = co_dim2event(cfg,data)
% CO_DIM2EVENT extracts event information from a specified dimension by creating events when a
% specified absolute threshold is reached across the time dimension. 
%
% INPUTS:
% cfg.FIELD.dim         = ['chan']
% cfg.FIELD.cell        = ['all']
% cfg.FIELD.select      = ['all']
% cfg.FIELD.threshold   = when the maximum absolute value crosses this threshold, an event will be
%                         created. If not specified, the threshold will be set to half the maxabs of
%                         the specified dimension selection.
% cfg.FIELD.offset      = [0] or array of offsets per selected dimension size. This value is added
%                         to the specified dimension before thresholding is performed.
% cfg.FIELD.samples     = [0 Inf] 2-element array containing minimum and maximum number of
%                         samples.
% cfg.FIELD.smooth      = [1] number of samples by which to smooth thresholded triggers for cases
%                         where a trigger might have multiple peaks close together.
% cfg.FIELD.evtvalue    = [cfg.FIELD.dim] String or numeric value to assign event.
%
% data
%
% OUTPUTS:
% data
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZ
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);

% Set defaults
for ii = 1:length(fields)
    if ~isfield(cfg.(fields{ii}),'dim'); cfg.(fields{ii}).dim = 'chan'; end;
    if ~isfield(cfg.(fields{ii}),'cell'); cfg.(fields{ii}).cell = 'all'; end;
    if ~isfield(cfg.(fields{ii}),'select'); cfg.(fields{ii}).select = 'all'; end;
    if ~isfield(cfg.(fields{ii}),'offset'); cfg.(fields{ii}).offset = 0; end;
    if ~isfield(cfg.(fields{ii}),'samples'); cfg.(fields{ii}).samples = [0 'Inf']; end;
    if ~isfield(cfg.(fields{ii}),'smooth'); cfg.(fields{ii}).smooth = 1; end;
    if ~isfield(cfg.(fields{ii}),'evtvalue'); cfg.(fields{ii}).evtvalue = cfg.(fields{ii}).dim; end;
end

% Select targeted dimension labels
data_sel = co_selectdim(cfg,data);

for ii = 1:length(fields)
    % Determine which cell(s) to operate on
    if ischar(cfg.(fields{ii}).cell) && strcmp(cfg.(fields{ii}).cell,'all')
        cellix = 1:length(data_sel.(fields{ii}));
    else
        cellix = cfg.(fields{ii}).cell;
    end
    
    for jj = cellix
        indata = data_sel.(fields{ii}){jj};
        intimedimix = find(strcmp('time',dim.(fields{ii})));
        
        assert(~isempty(intimedimix), '''time'' dimension is missing.');

        if intimedimix > 1   % Rearrange dimensions so time dim is 1
            indata = shiftdim(indata, intimedimix-1);
        end
        indatashiftsz = size(indata);
        
        indata = reshape(indata,indatashiftsz(1),prod(indatashiftsz(2:end)));   % Make indata 2-D
        indata = max(abs(indata+cfg.(fields{ii}).offset),[],2);
        
        % Set threshold cutoff to half of maxabs if threshold not configured
        if ~isfield(cfg.(fields{ii}),'threshold') || isempty(cfg.(fields{ii}).threshold)
            threshcutoff = max(indata)/2;
        else
            threshcutoff = cfg.(fields{ii}).threshold;
        end
        
        thresh = abs(indata) > threshcutoff;
        if cfg.(fields{ii}).smooth > 1
            thresh_smooth = thresh;
            for kk = 1:length(thresh)
                if thresh(kk); thresh_smooth(kk:min(kk+cfg.(fields{ii}).smooth,length(thresh)))=1; end;
            end
            thresh = thresh_smooth;
        end
        thresh = [0; thresh];
        negthresh = find(diff(thresh)<0);   % Trailing edge detection
        thresh = find(diff(thresh)>0);      % Leading edge detection
        for kk = 1:length(thresh)
            if ~isfield(data.event,fields{ii}) || length(data.event.(fields{ii}))<jj
                evtlen = 0;
            else
                evtlen = length(data.event.(fields{ii})(jj).sample);
            end
            
            if (cfg.(fields{ii}).samples(1) ~= 0 || cfg.(fields{ii}).samples(2) ~= Inf) && ...
                    length(negthresh) >= kk
                trigdur = negthresh(kk)-thresh(kk); % Trigger duration
                if trigdur < cfg.(fields{ii}).samples(1) || trigdur > cfg.(fields{ii}).samples(2)
                    continue;
                end
            end
            data.event.(fields{ii})(jj).sample(evtlen+1) = thresh(kk);
            data.event.(fields{ii})(jj).value{evtlen+1} = cfg.(fields{ii}).evtvalue;
        end
    end
end

data = co_sortevents([],data);

data = co_logcfg(cfg,data);
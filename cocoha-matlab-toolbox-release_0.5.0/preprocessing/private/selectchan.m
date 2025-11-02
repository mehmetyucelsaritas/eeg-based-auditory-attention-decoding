function data = selectchan(cfg,data,varargin)
% SELECTCHAN is an internal funtion for including/excluding channels from the data.
% This function is deprecated and will be replaced by CO_SELECTDIM.
%
% INPUTS
% cfg.FIELD.channels    = ['all']/{'all','-A1', ...}, which channels to include/exclude.
% data                  =
% varargin{1}           = FIELD to process (others will be ignored)
%
% OUTPUTS
% data                  =
%
% See also CO_PREPROCESSING
%
%
% Copyright 2015
% Daniel D.E. Wong, COCOHA Project, ENS/CNRS

warning('This function is deprecated and will be replaced by CO_SELECTDIM.')
fields = fieldnames(cfg);
if ~isempty(varargin); I = strmatch(varargin{1},fields); else I = 1:length(fields); end;
for ii = 1:I
    if ~iscell(cfg.(fields{ii}).channels)   % Convert to cell
        if isnumeric(cfg.(fields{ii}).channels); cfg.(fields{ii}).channels = strread(num2str(cfg.(fields{ii}).channels(:)'),'%s'); end;
        if isstr(cfg.(fields{ii}).channels); cfg.(fields{ii}).channels = {cfg.(fields{ii}).channels}; end;
    end
    
    % Locate chan dimension
    dim = strread(data.dim.(fields{ii}),'%s','delimiter','_');
    dimix = strmatch('chan',dim);
    
    selix = zeros(1,length(data.dim.chan.(fields{ii}))); % Start with nothing
    for jj = 1:length(cfg.(fields{ii}).channels)
        if dimix > 1    % Rearrange dimensions so chan is first dim
            data.(fields{ii}){jj} = shiftdim(data.(fields{ii}){jj}, dimix-1);
        end
        
        if cfg.(fields{ii}).channels{jj}(1) == '-'  % Remove channel
            selval = 0;
            cfg.(fields{ii}).channels{jj} = cfg.(fields{ii}).channels{jj}(2:end);
        else
            selval = 1;
        end
        if strcmp(cfg.(fields{ii}).channels{jj},'all')
            selix(:) = 1;
        else
            chix = strmatch(cfg.(fields{ii}).channels{jj},data.dim.chan.(fields{ii}),'exact');
            if ~isempty(chix); selix(chix) = selval; end
        end
    end
    
    for jj = 1:length(data.(fields{ii}))
        sz = size(data.(fields{ii}){jj});
        data.(fields{ii}){jj} = data.(fields{ii}){jj}(find(selix),:);
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},sz);
        if dimix > 1   % Rearrange dimensions to original order
            data.(fields{ii}){jj} = shiftdim(data.(fields{ii}){jj}, length(dim)-dimix+1);
        end
    end
    data.dim.chan.(fields{ii}) = data.dim.chan.(fields{ii})(find(selix));
end
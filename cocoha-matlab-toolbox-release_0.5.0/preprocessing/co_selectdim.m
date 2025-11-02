function data = co_selectdim(cfg,data)
% CO_SELECTDIM allows the for the removal of specified dimension indexes.
%
% INPUTS:
% cfg.FIELD.cell    = ['all'] / indexed array
% cfg.FIELD.dim     = name of dimension to select from (e.g. chan)
% cfg.FIELD.select  = ['all'] / indexed array / cell of labels - e.g. {'all','-Cz'}, etc. If the
%                     dimension labels consist of numbers e.g. {1, 2 ...}, then the select field
%                     must also consist of a cell of numbers; in which case the '-' operator cannot 
%                     be used. Non-existent indexes/labels will be ignored.
% data
%
% OUTPUTS:
% data
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU/Oticon, UCL, ETH Zurich
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);
for ii = 1:length(fields)
    assert(isfield(cfg.(fields{ii}),'dim'), ['Dimension not specified for ' fields{ii} '.']);
    if ~isfield(cfg.(fields{ii}),'cell'); cfg.(fields{ii}).cell = 'all'; end;
    if ~isfield(cfg.(fields{ii}),'select'); cfg.(fields{ii}).select = 'all'; end;
    
    %indim = strread(data.dim.(infield),'%s','delimiter','_');
    indimix = find(strcmp(cfg.(fields{ii}).dim,dim.(fields{ii})));
    if isempty(indimix); continue; end;     % cfg.FIELD.dim not found - handle gracefully
    
    % Select cells
    if ischar(cfg.(fields{ii}).cell)
        assert(strcmp(cfg.(fields{ii}).cell,'all'),['Unrecognized cell specification in cfg.' fields{ii} '.cell.']);
        cells = 1:length(data.(fields{ii}));
    else
        cells = intersect(cfg.(fields{ii}).cell,1:length(data.(fields{ii})));
    end
    
    % Handle string selection type
    if ischar(cfg.(fields{ii}).select)
        cfg.(fields{ii}).select = {cfg.(fields{ii}).select};
    end
    
    % Rearrange dimensions so dim is 1st
    cfgtmp = []; cfgtmp.(fields{ii}).shift = indimix-1; data = co_shiftdim(cfgtmp,data);
    
    for jj = cells
        % Handle numeric selection type
        if isnumeric(cfg.(fields{ii}).select)
            selectix = intersect(cfg.(fields{ii}).select, 1:size(data.(fields{ii}){jj},1));
        end
        
        % Handle cell selection type
        if iscell(cfg.(fields{ii}).select)
            selectix = [];
            for kk = 1:length(cfg.(fields{ii}).select)
                if isnumeric(cfg.(fields{ii}).select{kk})   % Selection consists of cell of numbers
                    selectix = union(selectix,find(cfg.(fields{ii}).select{kk} == ...
                        cell2mat(data.dim.(dim.(fields{ii}){indimix}).(fields{ii}){jj})));
                elseif strcmp(cfg.(fields{ii}).select{kk},'all')
                    selectix = 1:size(data.(fields{ii}){jj},1);
                else
                    strsel = cfg.(fields{ii}).select{kk};
                    if strsel(1) == '-'; strsel(1) = []; deselect = 1; else deselect = 0; end;
                    ix = find(strcmp(strsel,data.dim.(dim.(fields{ii}){indimix}).(fields{ii}){jj}));
                    if deselect; selectix = setdiff(selectix,ix); else; selectix = union(selectix,ix); end;
                end
            end
        end

        cfgtmp = []; cfgtmp.(fields{ii}).cell = jj; sz = co_size(cfgtmp,data); szrs = [sz,1];
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},szrs(1),prod(szrs(2:end)));
        data.(fields{ii}){jj} = data.(fields{ii}){jj}(selectix,:);
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},[length(selectix) szrs(2:end)]);
        
        
        % Modify dim axis labels
        if isfield(data.dim,cfg.(fields{ii}).dim) && isfield(data.dim.(cfg.(fields{ii}).dim),fields{ii})
            data.dim.(cfg.(fields{ii}).dim).(fields{ii}){jj} = data.dim.(cfg.(fields{ii}).dim).(fields{ii}){jj}(selectix);
        end
        
        % Handle events if dim is time
        selectixdiff = diff([0; selectix(:)]);
        if strcmp(cfg.(fields{ii}).dim,'time') && isfield(data.event,fields{ii})
            [~,keepix] = intersect(data.event.(fields{ii})(jj).sample,selectix);
            data.event.(fields{ii})(jj).sample = data.event.(fields{ii})(jj).sample(keepix);
            data.event.(fields{ii})(jj).value = data.event.(fields{ii})(jj).value(keepix);
            for kk = 1:length(selectix)
                ix = find(data.event.(fields{ii})(jj).sample >= selectix(kk));
                data.event.(fields{ii})(jj).sample(ix) = data.event.(fields{ii})(jj).sample(ix) ...
                    - selectixdiff(kk) + 1;
            end
        end
    end
    
    % Restore dimension order
    cfgtmp = []; cfgtmp.(fields{ii}).shift = 1-indimix; data = co_shiftdim(cfgtmp,data);
end

% Save cfg settings for future reference
data = co_logcfg(cfg,data);
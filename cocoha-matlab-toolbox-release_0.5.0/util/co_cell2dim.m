function data = co_cell2dim(cfg,data)
% CO_CELL2DIM collapses data cells into a new dimension. If there are dimension values, any that are
% not present across all cells will be discarded.
%
% INPUTS:
% cfg.FIELD.newdim  = new dimension name.
% cfg.FIELD.dimval  = (optional) cell array of dimension values.
%
% data
%
% OUTPUTS:
% data
%
% See also: CO_DIM2CELL
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZ
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);

for ii = 1:length(fields)
    assert(isfield(cfg.(fields{ii}),'newdim'), ['cfg.' fields{ii} '.newdim not specified.']);
    assert(~any(strcmp(dim.(fields{ii}),cfg.(fields{ii}).newdim)), ['Dimension ' ...
        cfg.(fields{ii}).newdim ' already exists']);
    
    % Check if dimension is numeric - to handle dimension value intersect
    numericdim = struct();
    for jj = 1:length(dim.(fields{ii}))
        % Check if axis values exist for dimension jj
        if isfield(data.dim,dim.(fields{ii}){jj}) && isfield(data.dim.(dim.(fields{ii}){jj}),fields{ii})
            numericdim.(dim.(fields{ii}){jj}) = isnumeric(data.dim.(dim.(fields{ii}){jj}).(fields{ii}){1}{1});
            for kk = 2:length(data.(fields{ii}))
                assert(numericdim.(dim.(fields{ii}){jj}) == isnumeric(data.dim.(dim.(fields{ii}){jj}).(fields{ii}){kk}{1}), ...
                    'Dimension values should all be numeric or strings, not a mix.');
            end
            if numericdim.(dim.(fields{ii}){jj}) % Convert numeric dimension to string
                for kk = 1:length(data.(fields{ii}))
                    for mm = 1:length(data.dim.(dim.(fields{ii}){jj}).(fields{ii}){kk})
                        data.dim.(dim.(fields{ii}){jj}).(fields{ii}){kk}{mm} = num2str( ...
                            data.dim.(dim.(fields{ii}){jj}).(fields{ii}){kk}{mm});
                    end
                end
            end
        end
    end
    
    dimvalues = struct();                   % Axis values for each dim, for each cell
    for jj = 1:length(dim.(fields{ii}))     % Cycle through dimensions
        % Check if axis values exist for dimension jj
        if isfield(data.dim,dim.(fields{ii}){jj}) && isfield(data.dim.(dim.(fields{ii}){jj}),fields{ii})
            % Store dimension values for cell kk
            for kk = 1:length(data.(fields{ii}))
                dimvalues(kk).(dim.(fields{ii}){jj}) = data.dim.(dim.(fields{ii}){jj}).(fields{ii}){kk};
            end
        end
    end
    
    dimswithvals = fieldnames(dimvalues(1));
    for jj = 1:ndims(data.(fields{ii}))                         % Cycle through dims
        if any(strcmp(dim.(fields{ii}){jj},dimswithvals))       % Dimension has values
            dimvalset = ones(1,length(dimvalues(1).(dim.(fields{ii}){jj}))); % Whether dimension labels are same across cells
            for kk = 2:length(dimvalues)                        % Get intersection with all cells
                minLen = min(length(dimvalset),length(dimvalues(1).(dim.(fields{ii}){jj})));
                dimvalset(1:minLen) = dimvalset(1:minLen) & strcmp(dimvalues(1).(dim.(fields{ii}){jj})(1:minLen), ...
                    dimvalues(kk).(dim.(fields{ii}){jj})(1:minLen));
                dimvalset(minLen:end) = 0;
            end
            dimvalset = logical(dimvalset);
            
            assert(any(dimvalset), ['There are no intersecting dimension label values ' ...
                'across all cells for dimension: ' dim.(fields{ii}){jj}]);
            
            for kk = 1:length(data.(fields{ii}))
                data.(fields{ii}){kk} = data.(fields{ii}){kk}(dimvalset,:);
                
                % Remove dimension values not in dimvalset
                data.dim.(dim.(fields{ii}){jj}).(fields{ii}){kk} = ...
                    data.dim.(dim.(fields{ii}){jj}).(fields{ii}){kk}(dimvalset);
                
                % If dimension is 'time' then remove events with samples not in dimvalset
                if strcmp(dim.(fields{ii}){jj},'time') && isfield(data.event,fields{ii})
                    keepevtix = find(data.event.(fields{ii})(kk).sample == find(dimvalset));
                    data.event.(fields{ii})(kk).sample = data.event.(fields{ii})(kk).sample(keepevtix);
                    data.event.(fields{ii})(kk).value = data.event.(fields{ii})(kk).value(keepevtix);
                end
            end
        else
            minlen = size(data.(fields{ii}){1},1);
            for kk = 2:length(data.(fields{ii}))
                minlen = min(minlen,size(data.(fields{ii}){kk},1));
            end
            for kk = 1:length(data.(fields{ii}))
                data.(fields{ii}){kk} = data.(fields{ii}){kk}(1:minlen,:);
                
                % If dimension is 'time' then remove events with samples greater than minlen
                if strcmp(dim.(fields{ii}){jj},'time') && isfield(data.event,fields{ii})
                    keepevtix = find(data.event.(fields{ii})(kk).sample <= minlen);
                    data.event.(fields{ii})(kk).sample = data.event.(fields{ii})(kk).sample(keepevtix);
                    data.event.(fields{ii})(kk).value = data.event.(fields{ii})(kk).value(keepevtix);
                end
            end
        end
        
        % Shift dims so that dim jj+1 will be in front
        for kk = 1:length(data.(fields{ii}))
            data.(fields{ii}){kk} = shiftdim(data.(fields{ii}){kk},1);
        end
    end
    
    % Combine cells
    sz = size(data.(fields{ii}){1});
    for jj = 2:length(data.(fields{ii}))
        data.(fields{ii}){1} = [data.(fields{ii}){1}(:); data.(fields{ii}){jj}(:)];
    end
    sz = [sz(1:length(dim.(fields{ii}))) length(data.(fields{ii}))];
    data.(fields{ii}){1} = reshape(data.(fields{ii}){1},sz);
    data.(fields{ii}) = data.(fields{ii})(1);
    
    % Retain dimension values for first cell only
    for jj = 1:length(dimswithvals)
        data.dim.(dimswithvals{jj}).(fields{ii}) = data.dim.(dimswithvals{jj}).(fields{ii})(1);
        if numericdim.(dimswithvals{jj})
            for kk = 1:length(data.dim.(dimswithvals{jj}).(fields{ii}){1})
                data.dim.(dimswithvals{jj}).(fields{ii}){1}{kk} = ...
                    str2double(data.dim.(dimswithvals{jj}).(fields{ii}){1}{kk});
            end
        end
    end
    
    % Add new dimension
    data.dim.(fields{ii}) = [data.dim.(fields{ii}) '_' cfg.(fields{ii}).newdim];
    
    % Add new dimension labels if specified
    if isfield(cfg.(fields{ii}),'dimval')
        assert(iscell(cfg.(fields{ii}).dimval), ['cfg.' fields{ii} '.dimval must be a cell.']);
        assert(numel(cfg.(fields{ii}).dimval) == sz(end), ...
            ['Length of cfg.' fields{ii} '.dimval must be equal to the number of data cells.']);
        data.dim.(cfg.(fields{ii}).newdim).(fields{ii}){1} = cfg.(fields{ii}).dimval(:)';
    end
    
    % Combine events
    if isfield(data.event,fields{ii})
        for jj = 2:length(data.(fields{ii}))
            data.event.(fields{ii})(1).sample = [data.event.(fields{ii})(1).sample(:); ...
                data.event.(fields{ii})(jj).sample(:)];
            data.event.(fields{ii})(1).sample = [data.event.(fields{ii})(1).value(:); ...
                data.event.(fields{ii})(jj).value(:)];
        end
        data.event.(fields{ii}) = data.event.(fields{ii})(1);
        
        % Remove duplicate events
        [data.event.(fields{ii})(1).sample,unqix] = unique(data.event.(fields{ii})(1).sample);
        data.event.(fields{ii})(1).value = data.event.(fields{ii})(1).value(unqix);
        
        % Sort events
        [data.event.(fields{ii})(1).sample,sortix] = sort(data.event.(fields{ii})(1).sample);
        data.event.(fields{ii})(1).value = data.event.(fields{ii})(1).value(sortix);
    end
end

data = co_logcfg(cfg,data);
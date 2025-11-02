function data = co_dim2cell(cfg,data)
% CO_DIM2CELL splits a dimension for a specified cell into multiple cells. Dimension labels will be 
% removed for the specified dimension. All other cells will be removed.
%
% cfg.FIELDS.cell   = [1] datafield cell to be operated on.
% cfg.FIELDS.dim    = dimension to be converted to cells.
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU/Oticon, UCL, ETH Zurich
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);

for ii = 1:length(fields)
    if ~isfield(cfg.(fields{ii}),'cell'); cfg.(fields{ii}).cell = 1; end;
    
    % Remove other cells
    data.(fields{ii}) = data.(fields{ii})(cfg.(fields{ii}).cell);
    for jj = 1:length(dim.(fields{ii}))
        if isfield(data.dim,dim.(fields{ii}){jj}) && isfield(data.dim.(dim.(fields{ii}){jj}),fields{ii})
            data.dim.(dim.(fields{ii}){jj}).(fields{ii}) = ...
                data.dim.(dim.(fields{ii}){jj}).(fields{ii})(cfg.(fields{ii}).cell);
        end
    end
    if isfield(data.event,fields{ii}); data.event.(fields{ii}) = data.event.(fields{ii})(cfg.(fields{ii}).cell); end;
    
    % Shift dim to first dimension
    dimix = find(strcmp(cfg.(fields{ii}).dim, dim.(fields{ii})));
    cfgtmp = [];
    cfgtmp.(fields{ii}).shift = dimix-1;
    data = co_shiftdim(cfgtmp,data);
    
    % Copy dim to cells and squeeze out first dim
    sz = size(data.(fields{ii}){1});
    for jj = [2:sz(1) 1]
        data.(fields{ii}){jj} = data.(fields{ii}){1}(jj,:);
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},sz(2:end));
    end
    
    % Remove dim labels
    if isfield(data.dim,cfg.(fields{ii}).dim)
        if isfield(data.dim.(cfg.(fields{ii}).dim),fields{ii})
            data.dim.(cfg.(fields{ii}).dim) = rmfield(data.dim.(cfg.(fields{ii}).dim),fields{ii});
        end
        if isempty(fieldnames(data.dim.(cfg.(fields{ii}).dim)))
            data.dim = rmfield(data.dim,cfg.(fields{ii}).dim);
        end
    end
    
    % Remove dim from dimension descriptor
    dims = co_strsplit(data.dim.(fields{ii}),'_');
    data.dim.(fields{ii}) = [];
    for jj = 2:length(dims)
        data.dim.(fields{ii}) = [data.dim.(fields{ii}) dims{jj} '_'];
    end
    data.dim.(fields{ii})(end) = [];
    
    % Duplicate dimension label cells for new data cells (and remove ones from old cells)
    for jj = 2:length(dims)
        if isfield(data.dim,dims{jj}) && isfield(data.dim.(dims{jj}),fields{ii})
            data.dim.(dims{jj}).(fields{ii}) = data.dim.(dims{jj}).(fields{ii})(1);
            data.dim.(dims{jj}).(fields{ii}) = repmat(data.dim.(dims{jj}).(fields{ii}), ...
                1,length(data.(fields{ii})));
        end
    end
    
    % Shift dimensions back
    cfgtmp = [];
    cfgtmp.(fields{ii}).shift = -dimix+1;
    data = co_shiftdim(cfgtmp,data);
    
    % Copy events to all cells
    for jj = 2:length(data.(fields{ii}))
        data.event.(fields{ii})(jj) = data.event.(fields{ii})(1);
    end
end

data = co_logcfg(cfg,data);
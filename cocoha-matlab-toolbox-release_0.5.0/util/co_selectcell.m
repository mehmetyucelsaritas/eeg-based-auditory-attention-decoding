function data = co_selectcell(cfg,data)
% CO_SELECTCELL selects trials (cells) in a data field to keep.
%
% INPUTS:
% cfg.FIELD.cell    = An array of cells to keep. If not specified, all cells in the data field will
%                     be kept.
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
    data.(fields{ii}) = data.(fields{ii})(cfg.(fields{ii}).cell);
    
    % Take care of labels
    for jj = 1:length(dim.(fields{ii}))
        if isfield(data.dim,dim.(fields{ii}){jj}) && isfield(data.dim.(dim.(fields{ii}){jj}),fields{ii})
            data.dim.(dim.(fields{ii}){jj}).(fields{ii}) = ...
                data.dim.(dim.(fields{ii}){jj}).(fields{ii})(cfg.(fields{ii}).cell);
        end
    end
    
    % Take care of events
    if isfield(data.event,fields{ii})
        data.event.(fields{ii}) = data.event.(fields{ii})(cfg.(fields{ii}).cell);
    end
end
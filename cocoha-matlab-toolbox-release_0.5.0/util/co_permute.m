function data = co_permute(cfg,data)
% CO_PERMUTE rearranges the dimensions of the data.
%
% INPUTS:
% cfg.FIELD.permix  = permutation indexes.
%
% data
%
% OUTPUTS:
% data
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZH
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);

for ii = 1:length(fields)
    assert(length(cfg.(fields{ii}).permix) == length(dim.(fields{ii})), ...
        'Number of permutation indexes does not match the number of dimensions indicated in data.dim.')
    
    % Permute actual data
    for jj = 1:length(data.(fields{ii}))
        data.(fields{ii}){jj} = permute(data.(fields{ii}){jj},cfg.(fields{ii}).permix);
    end
    
    % Permute dimension order
    data.dim.(fields{ii}) = [];
    for jj = 1:length(dim.(fields{ii}))
        data.dim.(fields{ii}) = [data.dim.(fields{ii}) '_' dim.(fields{ii}){cfg.(fields{ii}).permix(jj)}];
    end
    data.dim.(fields{ii}) = data.dim.(fields{ii})(2:end);
end

data = co_logcfg(cfg,data); % Save cfg settings for future reference
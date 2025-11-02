function data = co_squeeze(cfg,data)
% CO_SQUEEZE removes singleton dimensions in the data, except for the 'time' dimension. The squeezed
% dimension(s) must be singleton across all cells.
%
% INPUTS:
% cfg.FIELD         = specify an empty fieldname to remove all singleton dimensions.
%           .dim    = cell array of specified dimensions to remove. Only singleton dimensions will
%                     be removed.
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
    timeix = find(strcmp('time',dim.(fields{ii})));
    nonsingleton = zeros(1,length(dim.(fields{ii})));
    sz = cell(1,length(data.(fields{ii})));
    for jj = 1:length(data.(fields{ii}))
        cfgtmp = []; cfgtmp.(fields{ii}).cell = jj; sz{jj} = co_size(cfgtmp,data);
        nonsingleton = nonsingleton | sz{jj}>1;
    end
    singleton = ~nonsingleton; singleton(timeix) = 0;
    
    % Select only target dimensions defined in cfg.FIELD.dim
    if isfield(cfg.(fields{ii}),'dim')
        tgtdims = zeros(1,length(dim.(fields{ii})));
        for jj = 1:length(cfg.(fields{ii}).dim)
            dimix = find(strcmp(cfg.(fields{ii}).dim{jj},dim.(fields{ii})));
            tgtdims(dimix) = 1;
        end
        singleton = singleton & tgtdims;
    end
    nonsingleton = ~singleton; nonsingleton = find(nonsingleton);
    singleton = find(singleton);
    
    % Squeeze by reshaping
    for jj = 1:length(data.(fields{ii}))
        newsz = sz{jj}(nonsingleton); if length(newsz)==1; newsz = [newsz, 1]; end;
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},newsz);
    end
    
    % Remove singleton dimension information
    for jj = 1:length(singleton)
        if isfield(data.dim,dim.(fields{ii}){singleton(jj)}) && ...
                isfield(data.dim.(dim.(fields{ii}){singleton(jj)}),fields{ii})
            data.dim.(dim.(fields{ii}){singleton(jj)}) = rmfield( ...
                data.dim.(dim.(fields{ii}){singleton(jj)}),fields{ii});
            if isempty(data.dim.(dim.(fields{ii}){singleton(jj)}))
                data.dim = rmfield(data.dim, dim.(fields{ii}){singleton(jj)});
            end
        end
    end
    
    % Update dimension descriptor
    data.dim.(fields{ii}) = [];
    for jj = 1:length(nonsingleton)
        data.dim.(fields{ii}) = [data.dim.(fields{ii}), '_', dim.(fields{ii}){nonsingleton(jj)}];
    end
    data.dim.(fields{ii})(1) = [];
end

data = co_logcfg(cfg,data); % Save cfg settings for future reference
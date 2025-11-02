function data = co_shiftdim(cfg,data)
% CO_SHIFTDIM shifts the dimensions of a specified data field to the left by a specified amount.
% Dimensions will loop around onto the right hand side.
%
% INPUTS:
% cfg.FIELD.shift   = number of leftward dimension-shifts. A negative number will shift to the
%                     right.
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
    if length(dim.(fields{ii})) == 1 | cfg.(fields{ii}).shift == 0; continue; end; % Nothing to shift
    
    % Shift actual data
    for jj = 1:length(data.(fields{ii}))
        permix = circshift(1:length(dim.(fields{ii})),[0,-cfg.(fields{ii}).shift]);
        data.(fields{ii}){jj} = permute(data.(fields{ii}){jj},permix);
        
%         if cfg.(fields{ii}).shift < 0
%             shift = ndims(data.(fields{ii}){jj})+cfg.(fields{ii}).shift;
%         else
%             shift = cfg.(fields{ii}).shift;
%         end
%         data.(fields{ii}){jj} = shiftdim(data.(fields{ii}){jj},shift);
    end
    
    % Shift dim field
    dim.(fields{ii}) = circshift(dim.(fields{ii})(:),[-cfg.(fields{ii}).shift,0]);
    data.dim.(fields{ii}) = [];
    for jj = 1:length(dim.(fields{ii}))
        data.dim.(fields{ii}) = [data.dim.(fields{ii}) '_' dim.(fields{ii}){jj}];
    end
    data.dim.(fields{ii}) = data.dim.(fields{ii})(2:end);
end

% Save cfg settings for future reference
data = co_logcfg(cfg,data);
function data = co_multismooth(cfg,data)
% CO_MULTISMOOTH applies multiple smoothing kernels to the data. This function is a wrapper for
% NT_MULTISMOOTH from the NoiseTools toolbox.
%
% INPUTS:
% cfg.FIELD.dim             = ['time'] dimension to which to apply smoothing.
% cfg.FIELD.kernelsizes     = sizes of smoothing kernels.
% cfg.FIELD.alignment       = ['left']/'center'/'right'
% cfg.FIELD.dimname         = ['smooth'] name of dimension created by smoothing kernels. This
%                             dimension will only be created when multiple smoothing kernel sizes
%                             are specified.
%
% data
%
% OUTPUTS:
% data
%
% See also: NT_MULTISMOOTH
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZH
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);

if ~exist('nt_multismooth','var')
    addpath(fullfile(fileparts(which('co_defaults')), 'external', 'NoiseTools'));
    addpath(fullfile(fileparts(which('co_defaults')), 'external', 'NoiseTools', 'COMPAT'));
end

for ii = 1:length(fields)
    % Set defaults
    if ~isfield(cfg.(fields{ii}),'dim'); cfg.(fields{ii}).dim = 'time'; end;
    if ~isfield(cfg.(fields{ii}),'alignment'); cfg.(fields{ii}).alignment = 'left'; end;
    if ~isfield(cfg.(fields{ii}),'dimname'); cfg.(fields{ii}).dimname = 'smooth'; end;
    
    switch cfg.(fields{ii}).alignment
        case 'left'
            alignment = -1;
        case 'center'
            alignment = 0;
        case 'right'
            alignment = 1;
        otherwise
            error('Unrecognized alignment option.');
    end
    
    % Make cfg.FIELD.dim the first dimension
    dimix = find(strcmp(cfg.(fields{ii}).dim,dim.(fields{ii})));
    cfgtmp = []; cfgtmp.(fields{ii}).shift = dimix-1; data = co_shiftdim(cfgtmp,data);
    
    % Store sizes for each cell
    sz = cell(1,length(data.(fields{ii})));
    for jj = 1:length(data.(fields{ii}))
        cfgtmp = []; cfgtmp.(fields{ii}).cell = jj; sz{jj} = co_size(cfgtmp,data);
    end
    
    for jj = 1:length(data.(fields{ii}))
        % Reshape data to 2D
        szrs = [sz{jj},1];
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},szrs(1),prod(szrs(2:end)));
        
        data.(fields{ii}){jj} = nt_multismooth(data.(fields{ii}){jj},...
            cfg.(fields{ii}).kernelsizes,alignment);
        
        % Reshape data to ND
        if length(cfg.(fields{ii}).kernelsizes) > 1
            szrs = [szrs(1), length(cfg.(fields{ii}).kernelsizes), szrs(2:end)];
        end
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},szrs);
    end
    
    % Add dimension information
    if length(cfg.(fields{ii}).kernelsizes) > 1
        dimshift = co_strsplit(data.dim.(fields{ii}),'_');
        data.dim.(fields{ii}) = [];
        data.dim.(cfg.(fields{ii}).dimname).(fields{ii}) = cell(1,length(data.(fields{ii})));
        for jj = 1:length(dimshift) % Dimension names
            data.dim.(fields{ii}) = [data.dim.(fields{ii}), dimshift{jj}, '_'];
            if jj == 1; data.dim.(fields{ii}) = [data.dim.(fields{ii}), cfg.(fields{ii}).dimname, '_']; end;
        end
        data.dim.(fields{ii})(end) = [];
        for jj = 1:length(data.(fields{ii}))    % Axis labels
            data.dim.(cfg.(fields{ii}).dimname).(fields{ii}){jj} = ...
                mat2cell(cfg.(fields{ii}).kernelsizes(:)',1,ones(1,numel(cfg.(fields{ii}).kernelsizes(:))));
        end
    end
    
    % Shift dimensions back to original order
    cfgtmp = []; cfgtmp.(fields{ii}).shift = 1-dimix; data = co_shiftdim(cfgtmp,data);
end

data = co_logcfg(cfg,data);     % Save cfg settings for future reference

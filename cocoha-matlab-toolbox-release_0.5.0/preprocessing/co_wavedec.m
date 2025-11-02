function [data,wletinfo] = co_wavedec(cfg,data)
% CO_WAVEDEC performs a multilevel 1-D wavelet decomposition on specified cells.
% 
% INPUTS:
% cfg.FIELD.dim         = ['time']. Dimension to operate on.
% cfg.FIELD.cells       = ['all']/indexes. Cells to be operated on.
% cfg.FIELD.levels      = [1] the number of levels to filter the data.
% cfg.FIELD.wavelet     = ['haar'] or another wavelet filter type.
% data
%
% OUTPUTS:
% data_out              = data fields containing the wavelet decomposition of the specified cells.
% wletinfo              = information on wavelet coefficients. wletinfo.FIELD.L{CELL} contains
%                         an array containing the starting indexes of each wavelet decomposition
%                         component. The components are ordered the same way as they are in WAVEDEC.
%                         The last element in the array is the original length of the data. Note
%                         that this array is not the same as the L array outputted by WAVEDEC, which
%                         contains the lengths of the components and not the starting indexes!
%                         This was done to facilitate the possible subsequent use of CO_SPLITDATA.
%                         wletinfo.FIELD.cells specifies what cells were operated on.
%                         wletinfo.FIELD.dim specifies which dimension was operated on.
%                         wletinfo.FIELD.wlet specifies which wavelet was used.
%                         
%
% See also: WFILTERS, INTERP1, WAVEREC
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU/Oticon, UCL, ETH Zurich
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);

wletinfo = [];
for ii = 1:length(fields)
    if ~isfield(cfg.(fields{ii}),'dim'); cfg.(fields{ii}).dim = 'time'; end;
    if ~isfield(cfg.(fields{ii}),'cells'); cfg.(fields{ii}).cells = 'all'; end;
    if strcmp(cfg.(fields{ii}).cells,'all')
        cellix = 1:length(data.(fields{ii}));
    else
        cellix = cfg.(fields{ii}).cells;
    end
    if ~isfield(cfg.(fields{ii}),'levels'); cfg.(fields{ii}).levels = 1; end; levels = cfg.(fields{ii}).levels;
    if ~isfield(cfg.(fields{ii}),'wavelet'); cfg.(fields{ii}).wavelet = 'haar'; end; wavelet = cfg.(fields{ii}).wavelet;
    
    % Add to wletinfo structure
    wletinfo.(fields{ii}).dim = cfg.(fields{ii}).dim;
    wletinfo.(fields{ii}).cells = cellix;
    wletinfo.(fields{ii}).wlet = cfg.(fields{ii}).wlet;
    
    % Shift selected dim to first dimension
    dimix = find(strcmp(cfg.(fields{ii}).dim,dim.(fields{ii})));
    cfgtmp = [];
    cfgtmp.(fields{ii}).shift = dimix-1;
    data = co_shiftdim(cfgtmp,data);
    
    for jj = cellix
        % Make data 2D
        sz = data.(fields{ii}){jj};
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},sz(1),prod(sz(2:end)));
        
        % Wavelet decomposition
        outdata = [];
        for kk = 1:size(data.(fields{ii}){jj},2)
            [C,L] = wavedec(data.(fields{ii}){jj},levels,wavelet);
            if isempty(outdata); outdata = zeros(length(C),size(data.(fields{ii}){jj},2)); end;
            outdata(:,kk) = C;
        end
        data.(fields{ii}){jj} = reshape(outdata,sz);    % Make data N-D again
        
        % Add to wletinfo structure
        wletinfo.(fields{ii}).L{jj} = [1; 1+cumsum(L(1:end-2)); L(end)];
        
        % Pad dimension labels
        if isfield(data.dim,cfg.(fields{ii}).dim) && isfield(data.dim.(cfg.(fields{ii}).dim),fields{ii}) && ...
                length(data.dim.(cfg.(fields{ii}).dim).(fields{ii})) >= jj
            padlen = size(data_out.(fields{ii}){1},jj) - L(end);
            data.dim.(cfg.(fields{ii}).dim).(fields{ii}){jj}(end+1:end+padlen) = ...
                data.dim.(cfg.(fields{ii}).dim).(fields{ii}){jj}(end);
        end
    end
    
    % Restore original dimension order
    cfgtmp = [];
    cfgtmp.(fields{ii}).shift = -dimix+1;
    data = co_shiftdim(cfgtmp,data);
end

data = co_logcfg(cfg,data);
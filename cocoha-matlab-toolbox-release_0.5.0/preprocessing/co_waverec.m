function data = co_waverec(cfg,data)
% CO_WAVEREC reconstructs the original signal from data decomposed using CO_WAVEDEC.
%
% INPUTS:
% cfg.wletinfo    = the wletinfo structure returned by CO_WAVEDEC.
% data
%
% OUTPUTS:
% data
%
%
% See also: CO_WAVEDEC
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU/Oticon, UCL, ETH Zurich
% Author(s): Daniel D.E. Wong

wletinfo = cfg.wletinfo;
cfg = rmfield(cfg,'wletinfo');

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg.wletinfo);  % Modified from standard line - dims are provided in wletinfo

for ii = 1:length(fields)
    wletdim = wletinfo.(fields{ii}).dim;
    dimix = find(strcmp(dim.(fields{ii}),wletdim));
    cellix = wletinfo.(fields{ii}).cells;
    
    % Shift dimensions so wletdim is first
    cfgtmp = [];
    cfgtmp.(fields{ii}).shift = dimix-1;
    data = co_shiftdim(cfg,data);
    
    for jj = cellix
        % Prepare L vector for waverec
        L = wletinfo.(fields{ii}).L{jj};
        L(1:end-2) = L(2:end-1)-L(1:end-2);
        L(end-1) = size(data.(fields{ii}){jj},1) - L(end-1) + 1;
        
        % Make data 2D
        sz = size(data.(fields{ii}){jj});
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj}, sz(1), prod(sz(2:end)));
        
        % Wavelet reconstruction
        data.(fields{ii}){jj} = waverec(data.(fields{ii}){jj},L,wletinfo.wlet);
        
        % Restore data to N-D
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj}, sz);
        
        % Make sure dim labels are trimmed to size
        if isfield(data.dim,wletdim) && isfield(data.dim.(wletdim),fields{ii}) && ...
                length(data.dim.(wletdim).(fields{ii})) >= jj && ~isempty(data.dim.(wletdim).(fields{ii}){jj})
            data.dim.(wletdim).(fields{ii}){jj} = data.dim.(wletdim).(fields{ii}){jj}(1:sz(1));
        end
    end
    
    % Restore original dimension order
    cfgtmp = [];
    cfgtmp.(fields{ii}).shift = -dimix+1;
    data = co_shiftdim(cfg,data);
end
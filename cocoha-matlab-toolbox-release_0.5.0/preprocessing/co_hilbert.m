function data = co_hilbert(cfg,data)
% Performs a Hilbert transform on the data, with the option of adding the transformation as an
% additional dimension.
%
% cfg.FIELD.dim     = ['time'] dimension on which to operate.
% cfg.FIELD.op      = ['abs']/'complex'/'real'/'imag'/'absreal'/'absimag'/'angle'
% cfg.FIELD.newdim  = [], name of new dimension in which to store Hilbert transform. If no dimension
%                     is specified, the function will overwrite the original data.
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZH
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);

for ii = 1:length(fields)
    % Set defaults
    if ~isfield(cfg.(fields{ii}),'dim'); cfg.(fields{ii}).dim = 'time'; end;
    if ~isfield(cfg.(fields{ii}),'op'); cfg.(fields{ii}).op = 'abs'; end;
    if ~isfield(cfg.(fields{ii}),'newdim'); cfg.(fields{ii}).newdim = []; end;
    
    % Set operating dim to first position
    dimix = find(strcmp(cfg.(fields{ii}).dim,dim.(fields{ii})));
    cfgtmp = []; cfgtmp.(fields{ii}).shift = dimix-1; data = co_shiftdim(cfgtmp,data);
    
    % Get sizes for all cells
    sz = cell(1,length(data.(fields{ii})));
    for jj = 1:length(data.(fields{ii}))
        cfgtmp = []; cfgtmp.(fields{ii}).cell = jj; sz{jj} = co_size(cfgtmp,data);
    end
    
    for jj = 1:length(data.(fields{ii}))
        % Reshape data to 2D
        szrs = [sz{jj},1];  % szrs is only used for reshaping to 2D in case data has only 1D
        datatmp = reshape(data.(fields{ii}){jj},szrs(1),prod(szrs(2:end)));
        
        h = hilbert(datatmp);
        
        switch cfg.(fields{ii}).op
            case 'abs'
                h = abs(h);
            case 'complex'
                % No change to h
            case  'real'
                h = real(h);
            case 'imag'
                h = imag(h);
            case 'absreal'
                h = abs(real(h));
            case 'imagreal'
                h = imag(real(h));
            case 'angle'
                h = atan(real(h)./imag(h));
            otherwise
                error('Unrecognized op.');
        end
        
        if isempty(cfg.(fields{ii}).newdim)     % Overwrite data
            datatmp = h;
        else                                    % Insert as dimension
            datatmp = [datatmp, h];
            sz{jj} = [sz{jj},2];
        end
        
        % Reshape data to ND
        data.(fields{ii}){jj} = reshape(datatmp,sz{jj});
    end
    
    if ~isempty(cfg.(fields{ii}).newdim)
        data.dim.(fields{ii}) = [data.dim.(fields{ii}) '_' cfg.(fields{ii}).newdim];
    end
    
    % Set operating dimension back to original position
    cfgtmp = []; cfgtmp.(fields{ii}).shift = 1-dimix; data = co_shiftdim(cfgtmp,data);
end
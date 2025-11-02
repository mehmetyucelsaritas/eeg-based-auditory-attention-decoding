function data = co_diff(cfg,data)
% Differences the data along a specified dimension. Operation is applied to all cells. Only 1st
% order differencing is provided at this point.
%
% INPUTS:
% cfg.FIELD.dim         = ['time']
% cfg.FIELD.op          = ['none']/'abs'/'pos'/'neg'. Additional operation to perform on the
%                         differenced data. 'abs' will compute the absolute value. 'pos' will accept
%                         only positive values (setting negative values to zero). 'neg' will accept
%                         only negative values (setting positive values to zero).
% cfg.FIELD.newdim      = ['no']/'yes' whether to replace data with differenced data, or add
%                         differenced data as a new dimension.
% cfg.FIELD.diffdimname = ['diff'] name of dimension to add if cfg.FIELD.newdim is set to 'yes'.
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
    % Set defaults
    if ~isfield(cfg.(fields{ii}),'dim'); cfg.(fields{ii}).dim = 'time'; end;
    if ~isfield(cfg.(fields{ii}),'op'); cfg.(fields{ii}).op = 'none'; end;
    if ~isfield(cfg.(fields{ii}),'newdim'); cfg.(fields{ii}).newdim = 'no'; end;
    if ~isfield(cfg.(fields{ii}),'diffdimname'); cfg.(fields{ii}).diffdimname = 'diff'; end;
    
    % Set differenced dimension as first dimension
    dimix = find(strcmp(cfg.(fields{ii}).dim,dim.(fields{ii})));
    cfgtmp = []; cfgtmp.(fields{ii}).shift = dimix-1; data = co_shiftdim(cfgtmp,data);
    
    % Get sizes for all cells
    sz = cell(1,length(data.(fields{ii})));
    for jj = 1:length(data.(fields{ii}))
        cfgtmp = []; cfgtmp.(fields{ii}).cell = jj; sz{jj} = co_size(cfgtmp,data);
    end
    
    for jj = 1:length(data.(fields{ii}))
        % Reshape data to 2D
%         sz = size(data.(fields{ii}){jj});
        szrs = [sz{jj},1];  % szrs is only used for reshaping to 2D in case data has only 1D
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},szrs(1),prod(szrs(2:end)));
        
        diffdata = diff(data.(fields{ii}){jj},[],1);
        diffdata = [zeros(1,size(diffdata,2)); diffdata];
        
        switch cfg.(fields{ii}).op
            case 'none'
                % Do nothing
            case 'abs'
                diffdata = abs(diffdata);
            case 'pos'
                diffdata(diffdata<0) = 0;
            case 'neg'
                diffdata(diffdata>0) = 0;
            otherwise
                error(['Unrecognized cfg.' fields{ii} '.op argument.']);
        end
        
        if strcmp(cfg.(fields{ii}).newdim,'yes')
            data.(fields{ii}){jj} = [data.(fields{ii}){jj}, diffdata];
            sz{jj} = [sz{jj}, 2];
        else
            data.(fields{ii}){jj} = diffdata;
        end
        
        % Reshape data to ND
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},sz{jj});
    end
    
    % Create diff dimension if needed (no sense resetting dimension order in this case)
    if strcmp(cfg.(fields{ii}).newdim,'yes')    % Insert diff dimension if needed
        data.dim.(fields{ii}) = [data.dim.(fields{ii}), '_', cfg.(fields{ii}).diffdimname];
    else    % Reset dimension order
        cfgtmp = []; cfgtmp.(fields{ii}).shift = 1-dimix; data = co_shiftdim(cfgtmp,data);
    end

end

data = co_logcfg(cfg,data);     % Save cfg settings for future reference
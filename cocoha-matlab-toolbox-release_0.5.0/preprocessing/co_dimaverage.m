function data = co_dimaverage(cfg,data)
% CO_DIMAVERAGE computes the average (or sum) across a specified dimension.
%
% INPUTS
% cfg.FIELD.dimlabel    = ['chan'] dimension over which average will be taken.
% cfg.FIELD.dimrange    = ['all'] / cell array of dimension labels / indexes used for the dimension
%                         average.
% cfg.FIELD.weights     = [] weights by which to multiply data dimension. If dimension size across
%                         cells in data is the same, a vector can be specified; otherwise a cell
%                         array of vectors should be used. Note that CO_DIMAVERAGE will not
%                         normalize the weights. This is for the user to do.
% cfg.sum               = ['no']/'yes' compute sum instead of the mean.
% data
%
% OUTPUTS
% data
%
% See also: CO_SELECTDIM
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZH
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data,'sum');
fields = fieldnames(cfg);

for ii = 1:length(fields)
    % Set defaults
    if ~isfield(cfg.(fields{ii}),'dimlabel'); cfg.(fields{ii}).dimlabel = 'chan'; end;
    if ~isfield(cfg.(fields{ii}),'dimrange'); cfg.(fields{ii}).dimrange = 'all'; end;
    if ~isfield(cfg.(fields{ii}),'weights'); cfg.(fields{ii}).weights = []; end;
    if ~isfield(cfg,'sum'); cfg.sum = 'no'; end;
    dimix = strmatch(cfg.(fields{ii}).dimlabel,dim.(fields{ii}));
    
%     if isnumeric(cfg.(fields{ii}).dimrange)
%         dimrange = cfg.(fields{ii}).dimrange;
%     elseif strcmp(cfg.(fields{ii}).dimrange,'all')
%         dimrange = 1:size(data.(fields{ii}),dimix);
%     else
%         error(['Invalid cfg.' fields{ii} '.dimrange']);
%     end
    cfgtmp = [];
    cfgtmp.(fields{ii}).dim = cfg.(fields{ii}).dimlabel;
    cfgtmp.(fields{ii}).select = cfg.(fields{ii}).dimrange;
    data = co_selectdim(cfgtmp,data);
    
    for jj = 1:length(data.(fields{ii}))
        if dimix > 1   % Rearrange dimensions so dimdim is 1
            data.(fields{ii}){jj} = shiftdim(data.(fields{ii}){jj}, dimix-1);
        end
        szshift = size(data.(fields{ii}){jj});
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},szshift(1),prod(szshift(2:end)));
        
        % Set weights
        if isempty(cfg.(fields{ii}).weights)
            wts = ones(size(data.(fields{ii}){jj},1),1);
        elseif iscell(cfg.(fields{ii}).weights)
            wts = cfg.(fields{ii}).weights{jj}(:);
        else
            wts = cfg.(fields{ii}).weights(:);
        end

        %data.(fields{ii}){jj} = data.(fields{ii}){jj}(dimrange,:);
        
        data.(fields{ii}){jj} = data.(fields{ii}){jj} .* ...
            repmat(wts,1,size(data.(fields{ii}){jj},2));
        
        if strcmp(cfg.sum,'yes')
            data.(fields{ii}){jj} = sum(data.(fields{ii}){jj},1);
        else
            data.(fields{ii}){jj} = mean(data.(fields{ii}){jj},1);
        end
        

        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj}, [1, szshift(2:end)]);
        if dimix > 1   % Rearrange dimensions to original order
            data.(fields{ii}){jj} = shiftdim(data.(fields{ii}){jj}, length(dim.(fields{ii}))-dimix+1);
        end
        
        data.dim.(cfg.(fields{ii}).dimlabel).(fields{ii}){jj} = {'avg'};
    end
end

data = co_logcfg(cfg,data);
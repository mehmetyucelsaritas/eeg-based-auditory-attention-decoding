function data = co_auditoryfilterbank(cfg,data)
% CO_AUDITORYFILTERBANK applies a gammatone filter bank to the specified data field.
%
% INPUTS
% cfg.FIELD.dim         = ['time'] dimension on which to perform filter operation.
% cfg.FIELD.flow        = [80]
% cfg.FIELD.fhigh       = [8000]
%
% See also AUDITORYFILTERBANK
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZH
% Author(s): Daniel D.E. Wong

if ~exist('amtstart','file')
    addpath(fullfile(fileparts(which('co_defaults')), 'external', 'amtoolbox'));
    amtstart;
end

dim = co_checkdata(cfg,data);

fields = fieldnames(cfg);
for ii = 1:length(fields)
%     % Check dimensions
%     if strcmp(data.dim.(fields{ii}),'chan_time')
%         for jj = 1:length(data.(fields{ii}))
%             data.(fields{ii}){jj} = data.(fields{ii}){jj}';
%         end
%         data.dim.(fields{ii}){jj} = 'time_chan';
%     end
%     assert(strcmp(data.dim.(fields{ii}),'time_chan'), ['dim.' fields{ii} ' should be time_chan.']);

    % Set cfg defaults
    if ~isfield(cfg.(fields{ii}),'flow'); cfg.(fields{ii}).flow = 80; end;
    if ~isfield(cfg.(fields{ii}),'fhigh'); cfg.(fields{ii}).fhigh = 8000; end;
    
    % Set cfg.FIELDS.dim to first dimension
    dimix = find(strcmp(cfg.(fields{ii}).dim,dim.(fields{ii})));
    cfgtmp = []; cfgtmp.(fields{ii}).shift = dimix-1;
    data = co_shiftdim(cfgtmp,data);
    
    % Obtain sizes for all cells
    sz = cell(1,length(data.(fields{ii})));
    for jj = 1:length(data.(fields{ii}))
        cfgtmp = []; cfgtmp.(fields{ii}).cell = jj; sz{jj} = co_size(cfgtmp,data);
    end
    
    % Apply filter bank
    for jj = 1:length(data.(fields{ii}))
        % Reshape data to 2D
        szrs = [sz{jj} 1];  % Used only for reshaping to 2D
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},szrs(1),prod(szrs(2:end)));
        
        datatmp = [];
        for kk = 1:size(data.(fields{ii}){jj},2)
            [outsig,fc] = auditoryfilterbank(data.(fields{ii}){jj}(:,kk),data.fsample.(fields{ii}),'flow',cfg.(fields{ii}).flow,'fhigh',cfg.(fields{ii}).fhigh);
            if isempty(datatmp); datatmp= zeros([length(fc) size(data.(fields{ii}){jj})]); end;
            datatmp(:,:,kk) = outsig';
        end
        
        data.(fields{ii}){jj} = datatmp;
        
        % Reshape data to ND
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},[length(fc), sz{jj}]);
        
        fc = fc(:); data.dim.freq.(fields{ii}){jj} = mat2cell(fc,ones(1,size(fc,1),1))';
    end
    data.dim.(fields{ii}) = ['freq_', data.dim.(fields{ii})];   % Won't bother shifting dimensions back
end

% Save cfg settings for future reference
data = co_logcfg(cfg,data);
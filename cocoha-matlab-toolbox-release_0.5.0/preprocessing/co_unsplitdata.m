function data = co_unsplitdata(cfg,data)
% CO_UNSPLITDATA concatenates cells (trials) of a data field. This essentially reverses the data
% splitting done by CO_SPLITDATA.
%
% INPUTS:
% cfg.FIELD.dim     = ['time'] dimension to concatenate/unsplit.
% data              = data that was split using CO_SPLITDATA, or has equal sizes for dimensions
%                     other than the dimension being concatenated. 
%
% OUTPUTS:
% data              = data restored to pre-split dimensions.
%
% See also: CO_SPLITDATA
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZH
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);
for ii = 1:length(fields)
%     % Check event samples for all cells begin with a 1
%     assert(isfield(data.event,fields{ii}),['Absent event data for ' fields{ii}]);
%     for jj = 1:length(data.event.(fields{ii}))
%         assert(any(data.event.(fields{ii})(jj).sample(1)==1),['Invalid event information for ' fields{ii} '. Events for split data should contain a 1 in the sample field.']);
%     end
    
    if ~isfield(cfg.(fields{ii}),'dim'); cfg.(fields{ii}).dim = 'time'; end;
    dimname = cfg.(fields{ii}).dim;
    
    dimix = find(strcmp(dimname,dim.(fields{ii})));
    cfgtmp = []; cfgtmp.(fields{ii}).shift = dimix-1; data = co_shiftdim(cfgtmp,data);
%     szshift1 = zeros(1,ndims(data.(fields{ii}){jj}));     % Used for check that dim lengths align

    szshift = cell(1,length(data.(fields{ii})));
    for jj = 1:length(data.(fields{ii}))
        cfgtmp = []; cfgtmp.(fields{ii}).cell = jj; szshift{jj} = co_size(cfgtmp,data);
    end
    
    for jj = 1:length(data.(fields{ii}))
%         if dimix > 1    % Set dim idx to 1
%             data.(fields{ii}){jj} = shiftdim(data.(fields{ii}){jj}, dimix-1);
%         end
%         if jj == 1; szshift1 = size(data.(fields{ii}){jj}); end;
        
        szshift1rs = [szshift{jj},1]; szshiftjjrs = [szshift{jj},1];
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},szshiftjjrs(1),prod(szshiftjjrs(2:end))); % make data 2d
        if jj > 1
            assert(all(szshiftjjrs(2:end)==szshift1rs(2:end)), ...
                ['Dimensions of data.' fields{ii} '.{' num2str(jj) '} are not suitable for concatenating with data.' fields{ii} '.{1}.']);
            
            % Handle events
            if strcmp(dimname,'time') && isfield(data.event,fields{ii})
                data.event.(fields{ii})(1).sample = [data.event.(fields{ii})(1).sample(:); data.event.(fields{ii})(jj).sample(:)+size(data.(fields{ii}){1},1)];
                data.event.(fields{ii})(1).value = [data.event.(fields{ii})(1).value(:); data.event.(fields{ii})(jj).value(:)];
            end
            
            % Handle dims
            if isfield(data.dim,dimname) && isfield(data.dim.(dimname),fields{ii}) && ...
                    ~isempty(data.dim.(dimname).(fields{ii}){jj})
                data.dim.(dimname).(fields{ii}){1} = [data.dim.(dimname).(fields{ii}){1}(:); data.dim.(dimname).(fields{ii}){jj}(:)];
            end
            
            % Concatenate
            data.(fields{ii}){1} = [data.(fields{ii}){1}; data.(fields{ii}){jj}];
            data.(fields{ii}){jj} = []; % Save space
        end
    end
    
    data.(fields{ii}){1} = reshape(data.(fields{ii}){1},[size(data.(fields{ii}){1},1) szshift1rs(2:end)]);
    data.(fields{ii})(2:end) = [];
    
%     if dimix > 1   % Rearrange dimensions to original order
%         data.(fields{ii}){1} = shiftdim(data.(fields{ii}){1}, length(dim.(fields{ii}))-dimix+1);
%     end
    
    % Remove copied events
    if strcmp(dimname,'time') && isfield(data.event,fields{ii})
        data.event.(fields{ii})(2:end) = [];
        data.event.(fields{ii})(2:end) = [];
    end
    
    % Remove copied dims
    for jj = 1:length(dim.(fields{ii}))
        if isfield(data.dim,dim.(fields{ii}){jj}) && ...
                isfield(data.dim.(dim.(fields{ii}){jj}),fields{ii}) && ...
                length(data.dim.(dim.(fields{ii}){jj}).(fields{ii}))>1
            data.dim.(dim.(fields{ii}){jj}).(fields{ii})(2:end) = [];
        end
    end
    
    % Rearrange dimensions to original order
    cfgtmp = []; cfgtmp.(fields{ii}).shift = 1-dimix; data = co_shiftdim(cfgtmp,data);
end
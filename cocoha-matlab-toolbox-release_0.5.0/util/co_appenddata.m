function data = co_appenddata(cfg,data,varargin)
% CO_APPENDDATA appends datasets along the specified dimension. All other dimensions must be equal.
% If dimension labels are present, those belonging to the dimension on which the append operation is
% being done must be unique (i.e. not replicated between datasets). Those belonging to the other
% dimensions must be the same. All data structures must contain at least the same number of trials
% (data cells) as the first. If concatenated dimension is time, events will be appended as well.
%
% INPUTS:
% cfg.FIELD.dim             = ['time'] the dimension to be concatenated. If unspecified, data will
%                             be appended as trials (cells).
% data                      = the first data structure, or a cell of data structures. If this is a
%                             cell, all elements will be appended to the first.
% varargin                  = data structures to be appended.
%
% OUTPUTS:
% data                      = a data structure containing the appended inputs.
%
% See also: CO_PREPROCESSING, CO_SPLITDATA and CO_UNSPLITDATA
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZH
% Author(s): Daniel D.E. Wong

% Turn data and varargin into cell array of data structures
if ~iscell(data)
    datatmp{1} = data;
    data = datatmp;
end
for ii = 1:length(varargin)
    data{ii+1} = varargin{ii};
end

dim = co_checkdata(cfg,data{1});
fields = fieldnames(cfg);

for ii = 1:length(fields)
%     if ~isfield(cfg.(fields{ii}),'dim'); cfg.(fields{ii}).dim = 'time'; end;
    
    if ~isfield(cfg.(fields{ii}),'dim')   % Append data as cells
        for jj = 1:length(data)
            if jj > 1
                assert(data{1}.fsample.(fields{ii}) == data{jj}.fsample.(fields{ii}), ...
                    ['Sampling rates for datasets 1 and ' num2str(jj) ' are different.']);
                assert(strcmp(data{1}.dim.(fields{ii}),data{jj}.dim.(fields{ii})), ...
                    ['Dimension names for datasets 1 and ' num2str(jj) ' are different.']);
                
                ntrials1 = length(data{1}.(fields{ii}));   % Number of trials (cells) in data 1
                ntrialsjj = length(data{jj}.(fields{ii}));  % Number of trials (cells) in data jj
                data{1}.(fields{ii})(ntrials1+1:ntrials1+ntrialsjj) = data{jj}.(fields{ii});
                
                % Handle dimension labels
                for kk = 1:length(dim.(fields{ii}))
                    if isfield(data{1}.dim,dim.(fields{ii}){kk}) && isfield(data{1}.dim.(dim.(fields{ii}){kk}),fields{ii})
                        if ~isfield(data{jj}.dim,dim.(fields{ii}){kk}) || ~isfield(data{jj}.dim.(dim.(fields{ii}){kk}),fields{ii})
                            fprintf('Dimension labels for dimension %s not found in dataset %i. Removing dimension labels altogether.\n', ...
                                fields{ii}, jj);
                            data{1}.dim.(dim.(fields{ii}){kk}) = ...
                                rmfield(data{1}.dim.(dim.(fields{ii}){kk}),fields{ii});
                        else
                            data{1}.dim.(dim.(fields{ii}){kk}).(fields{ii})(ntrials1+1:ntrials1+ntrialsjj) = ...
                                data{jj}.dim.(dim.(fields{ii}){kk}).(fields{ii});
                        end
                    end

                end
                
                % Handle events
                if isfield(data{jj}.event,fields{ii})
                    data{1}.event.(fields{ii})(ntrials1+1:ntrials1+ntrialsjj) = ...
                        data{jj}.event.(fields{ii});
                end
            end
        end
        continue;
    end
    
    % Append data along specified dimension
    sz = cell(1,length(data));  % Store data sizes
    for jj = 1:length(data)
        % Make dim the first dimension
        dimix = find(strcmp(cfg.(fields{ii}).dim,dim.(fields{ii})));
        if jj==1; dimix1 = dimix; end;  % Used to restore dimension order
        cfgtmp = [];
        cfgtmp.(fields{ii}).shift = dimix-1;
        data{jj} = co_shiftdim(cfgtmp,data{jj});
        
        % Trial-independent assertions
        if jj > 1
            assert(length(data{jj}.(fields{ii})) >= length(data{1}.(fields{ii})), ...
                ['Trials in dataset ' num2str(jj) ' are fewer than in dataset 1.']);
            assert(strcmp(data{jj}.dim.(fields{ii}),data{jj}.dim.(fields{ii})), ...
                ['Dimensions between datasets 1 and ' num2str(jj) 'are different or have different ' ...
                'permutations. Unable to re-order correctly with shiftdim.']); % TODO: Create permutation fcn
            assert(data{1}.fsample.(fields{ii}) == data{jj}.fsample.(fields{ii}), ...
                    ['Sampling rates for datasets 1 and ' num2str(jj) ' are different.']);
        end
        
        sz{jj} = cell(1,length(data{jj}.(fields{ii})));
        for kk = 1:length(data{1}.(fields{ii}))
            sz{jj}{kk} = size(data{jj}.(fields{ii}){kk});

            % Trial-dependent assertions
            if jj > 1
                assert(ndims(data{jj}.(fields{ii}){kk}) == ndims(data{jj}.(fields{ii}){kk}), ...
                    ['Number of dimensions in dataset ' num2str(jj) ' is not equal to that in dataset 1.']);

                % Ensure same dim size for dims > 1
                assert(all(sz{jj}{kk}(2:end)==sz{jj}{kk}(2:end)), ['Dimension length mismatch between dataset ' ...
                    num2str(jj) ' and dataset 1.']);

                % Ensure dims with labels
                for mm = 1:length(dim.(fields{ii}))
                    if isfield(data{1}.dim,dim.(fields{ii}){mm}) && isfield(data{1}.dim.(dim.(fields{ii}){mm}),fields{ii})	% Dimension data exists for dim kk in data1
                        assert(isfield(data{jj}.dim,dim.(fields{ii}){mm}) && isfield(data{jj}.dim.(dim.(fields{ii}){mm}),fields{ii}), ...
                            ['Dimension data expected for dataset ' num2str(jj) ', dimension ''' ...
                            dim.(fields{ii}){mm} '''.']);

                        if strcmp(dim.(fields{ii}){mm},cfg.(fields{ii}).dim)    % Dimension kk being appended
                            % Ensure data jj has unique dimension labels
                            if ~unique(data{1}.dim.(dim.(fields{ii}){mm}).(fields{ii}){kk}, ...
                                    data{jj}.dim.(dim.(fields{ii}){mm}).(fields{ii}){kk})
                                warning(['Nonunique concatenated dimension labels for dataset ' ...
                                    num2str(jj) ' and dataset 1.']);
                            end
                        else
                            % Ensure data jj has the same dimension labels
                            if ~equals(data{1}.dim.(dim.(fields{ii}){mm}).(fields{ii}){kk}, ...
                                    data{jj}.dim.(dim.(fields{ii}){mm}).(fields{ii}){kk})
                                warning(['Dimension ''' dim.(fields{ii}){mm} ...
                                    ''' labels for dataset ' num2str(jj) ' and dataset 1 are not the same.']);
                            end
                        end
                    end
                end
            end
            
            % Concatenate data and dimension labels
            data{jj}.(fields{ii}){kk} = reshape(data{jj}.(fields{ii}){kk},sz{jj}{kk}(1),prod(sz{jj}{kk}(2:end)));
            if jj > 1
                data{1}.(fields{ii}){kk} = [data{1}.(fields{ii}){kk}; data{jj}.(fields{ii}){kk}];
                if isfield(data{1}.dim,cfg.(fields{ii}).dim) && isfield(data{1}.dim.(cfg.(fields{ii}).dim),fields{ii})
                    data{1}.dim.(cfg.(fields{ii}).dim).(fields{ii}){kk} = [ ...
                        data{1}.dim.(cfg.(fields{ii}).dim).(fields{ii}){kk}(:)' ...
                        data{jj}.dim.(cfg.(fields{ii}).dim).(fields{ii}){kk}(:)'];
                end
            end
        end
    end
    
    szfinal = zeros(length(data{1}.(fields{ii})));
    for jj = 1:length(data)                         % Tally final size for data1
        for kk = 1:length(data{1}.(fields{ii}))
            szfinal(kk) = szfinal(kk) + sz{jj}{kk}(1);
        end
    end
    for jj = 1:length(data{1}.(fields{ii}))         % Reshape data1
        data{1}.(fields{ii}){jj} = reshape(data{1}.(fields{ii}){jj}, [szfinal(1) sz{1}{jj}(2:end)]);
    end
    
    % Shift dims back for data1
    cfgtmp = [];
    cfgtmp.(fields{ii}).shift = -(dimix1-1);
    data{1} = co_shiftdim(cfgtmp,data{1});
    
    % Handle events if dimension is time
    if strcmp(cfg.(fields{ii}).dim,'time') && isfield(data{1}.event,fields{ii})
        evt_off = zeros(1,length(data{1}.(fields{ii})));	% Appended event offsets
        for jj = 2:length(data)
            if isfield(data{jj}.event,fields{ii})
                for kk = 1:length(data{1}.(fields{ii}))
                    evt_off(kk) = evt_off(kk) + sz{jj-1}{kk}(1);
                    data{1}.event.(fields{ii})(kk).sample = [data{1}.event.(fields{ii})(kk).sample(:); ...
                        data{jj}.event.(fields{ii})(kk).sample(:) + evt_off(kk)];
                    data{1}.event.(fields{ii})(kk).value = [data{1}.event.(fields{ii})(kk).value(:); ...
                        data{jj}.event.(fields{ii})(kk).value(:)];
                end
            end
        end
        
    end
end

data{1} = co_sortevents([],data{1});
cfgtmp = [];
for ii = 2:length(data)
    cfgtmp.datacfg{ii-1} = data{ii}.cfg;
end
data = data{1};

% Save cfg settings for future reference
data = co_logcfg(cfg,data);
data.cfg{end}.datacfg = cfgtmp.datacfg;


function u = unique(a,b)
% Unique function for cells a and b, regardless of string or not
u = true;
for ii = 1:length(a)
    for jj = 1:length(b)
        if length(a{ii})==length(b{jj}) && all(a{ii} == b{jj}); u = false; return; end;
    end
end

function e = equals(a,b)
% Determines if cells a and cells b are the same
e = true;
if length(a) ~= length(b); e = false; return; end;
for ii = 1:length(a)
    if ~all(a{ii} == b{ii}); e = false; return; end;
end
function [data_out] = co_statfeatures(cfg,data)
% CO_STATFEATURES computes univariate features across time for specified cells, as well as bivariate
% features along the non-time dimensions (e.g. channels or components) for each cell, and bivariate
% features between cells.
%
%
% INPUTS:
% cfg.FIELD.wavedec     = 'yes'/['no'] whether CO_WAVEDEC and CO_SPLITDATA have been called on the
%                         data. (TODO)
%
% cfg.FIELD.features    = {}, cell array with any of 'var','mean','skew','kurt'. Univariate
%                         features.
% cfg.FIELD.withincells = {}, Cell array with any of 'corrcoef','covar','coskew' and/or 'cokurt' 
%                         bivariate features. Computes bivariate features within cells, across 
%                         components.
%                         (e.g. component 1 for cell 1 x component 2 for cell 1).
% cfg.FIELD.withinix    = {}, cell array containing Nx2 array of component pairs over which to
%                         compute the bivariate features within-cells. If specified, there should be
%                         the same number of cells as data.FIELD. Otherwise, features will be 
%                         computed over all pairs.
% cfg.FIELD.acrosscells = {}, cell array with any of 'corrcoef','covar','coskew' and/or 'cokurt'  
%                         bivariate features. Computes bivariate features across cells, within
%                         components (e.g. component 1 for cell 1 x component 1 for cell 2). All
%                         cells must have the same number of time samples and components.
%
% Time dimension labels are determined by the event codes and effectively indicate the class to
% which the features at a given time sample belong. For instance, if there is an event at time
% sample 10 with a value of 1 and this condition lasts for 1000 time samples, we can set
% cfg.FIELD.triallen = 1000 to indicate that the sliding time windows used to compute the features
% should be aligned such that the first window begins at the start of each event (time sample 10 in
% this case) and should not be computed for times exceeding 1000 samples beyond the event onset.
%
% cfg.FIELD.triallen    = (scalar) length of trial for each condition. Sliding time windows used to
%                         compute features will be aligned to the beginning of each event code. If
%                         there are no event codes, a default code will be assumed at time sample 1,
%                         with an event value of 1.
% cfg.FIELD.timewindow  = (scalar) number of time samples in each time window over which the
%                         features are computed.
% cfg.FIELD.stepsize    = (scalar) number of time steps to shift the time window for each sample. If
%                         not specified, this will be equal to the value of cfg.FIELD.timewindow.
%
%
% data                  = data from which time-windowed features are to be extracted
%
%
% OUTPUTS:
% data_out              = 
% 
%
% See also: COMBNK to generate cfg.FIELD.withinix, and CO_DIMREDUCTION
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZ
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);
data_out = [];

for ii = 1:length(fields)
    % Asserts and default config
%     assert(isfield(cfg.(fields{ii}),'W') && length(cfg.(fields{ii}).W) == length(data.(fields{ii})), ...
%         ['cfg.' fields{ii} '.W has the wrong length.']);
    if ~isfield(cfg.(fields{ii}),'wavedec'); cfg.(fields{ii}).wavedec = 'no'; end;
    if ~isfield(cfg.(fields{ii}),'features'); cfg.(fields{ii}).features = {}; end;
    if ~isfield(cfg.(fields{ii}),'withincells'); cfg.(fields{ii}).withincells = {}; end;
    if ~isfield(cfg.(fields{ii}),'withinix'); cfg.(fields{ii}).withinix = cell(1,length(data.(fields{ii}))); end;
    if ~isfield(cfg.(fields{ii}),'acrosscells'); cfg.(fields{ii}).acrosscells = {}; end;
    assert(isfield(cfg.(fields{ii}),'triallen'), ['cfg.' fields{ii} '.triallen is undefined.']);
    assert(isfield(cfg.(fields{ii}),'timewindow'), ['cfg.' fields{ii} '.timewindow is undefined.']);
    assert(isfield(cfg.(fields{ii}),'stepsize'), ['cfg.' fields{ii} '.stepsize is undefined.']);
    
    % Add in an initial event if none exist
    for jj = 1:length(data.(fields{ii}))
        if ~isfield(data.event,fields{ii}) || length(data.event.(fields{ii}))<jj || ...
                ~isfield(data.event.(fields{ii})(jj),'sample') || isempty(data.event.(fields{ii})(jj).sample)
            data.event.(fields{ii})(jj).sample = 1;
            data.event.(fields{ii})(jj).value = {1};
        end
    end
    
    % Make time first dimension
    timeix = find(strcmp('time',dim.(fields{ii})));
    assert(~isempty(timeix), 'Time dimension does not exist.');
    cfgtmp = [];
    cfgtmp.(fields{ii}).shift = timeix-1;
    data = co_shiftdim(cfgtmp,data);
    
    % One more assert!
    if ~isempty(cfg.(fields{ii}).acrosscells)
        for jj = 2:length(data.(fields{ii}))
%             assert(size(cfg.(fields{ii}).W{jj},2)==size(cfg.(fields{ii}).W{jj},2), ...
%                 'All cells must have the same number of components if cfg.FIELD.acrosscells is used.');
            assert(all(size(data.(fields{ii}){jj}) == size(data.(fields{ii}){1})), ...
                ['All data cells must have the same dimension sizes if cfg.' fields{ii} '.acrosscells is used.']);
        end
    end
    
    data_out = [];
    
    for jj = 1:length(data.(fields{ii}))
        % Make data 2D and apply demixing matrix
        sz = size(data.(fields{ii}){jj});
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},sz(1),prod(sz(2:end)));
%         data.(fields{ii}){jj} = data.(fields{ii}){jj}*cfg.(fields{ii}).W{jj};
        
        if strcmp(cfg.(fields{ii}).wavedec,'yes')
            % Wavelet mode (TODO)
            error('Not yet implemented');
        end
    end
    
    for jj = 1:length(data.(fields{ii}))
        fprintf('[*] Field #%d of %d\n',jj,length(data.(fields{ii})));
        
        % Set trial lengths per event sample based on cfg
        if length(cfg.(fields{ii}).triallen) == 1
            triallen = cfg.(fields{ii}).triallen*ones(1,length(data.event.(fields{ii})(jj).sample));
        else
            triallen = cfg.(fields{ii}).triallen;
        end
        
        % Allocate space for time windows and features
        t_win = 0;
        for kk = 1:length(data.event.(fields{ii})(jj).sample)
            t_win = t_win + 1+floor((triallen(kk)-cfg.(fields{ii}).timewindow+1)/cfg.(fields{ii}).stepsize);
        end
        nfeat(1) = size(data.(fields{ii}){jj},2) * length(cfg.(fields{ii}).features);                                  % 1-variable features
        if isempty(cfg.(fields{ii}).withinix)                                                                           % Within cell features
            nfeat(2) = size(data.(fields{ii}){jj},2)*(size(data.(fields{ii}){jj},2)-1)/2 * length(cfg.(fields{ii}).withincells);
            if any(strcmp(cfg.(fields{ii}).withincells,'coskew'))   % Takes up 2 features
                nfeat(2) = nfeat(2) + size(data.(fields{ii}){jj},2)*(size(data.(fields{ii}){jj},2)-1)/2;
            end
            if any(strcmp(cfg.(fields{ii}).withincells,'coskew'))   % Takes up 4 features
                nfeat(2) = nfeat(2) + 3*size(data.(fields{ii}){jj},2)*(size(data.(fields{ii}){jj},2)-1)/2;
            end
        else
            nfeat(2) = size(cfg.(fields{ii}).withinix{jj},1) * length(cfg.(fields{ii}).withincells);
        end
        nfeat(3) = (length(data.(fields{ii}))-jj)*size(data.(fields{ii}){jj},2)*length(cfg.(fields{ii}).acrosscells);  % Across cell features
        if any(strcmp(cfg.(fields{ii}).acrosscells,'coskew'))       % Takes up 2 features
            nfeat(3) = nfeat(3) + (length(data.(fields{ii}))-jj)*size(data.(fields{ii}){jj},2);
        end
        if any(strcmp(cfg.(fields{ii}).acrosscells,'coskew'))       % Takes up 4 features
            nfeat(3) = nfeat(3) + 3*(length(data.(fields{ii}))-jj)*size(data.(fields{ii}){jj},2);
        end
        
        data_feat = zeros(t_win,sum(nfeat));
        time_labels = cell(1,t_win);
        feat_labels = cell(1,sum(nfeat));
        
        t_count = 0;
        for kk = 1:length(data.event.(fields{ii})(jj).sample)
            if mod(kk,10)==1; fprintf('\tProcessing event #%d of %d...\n',kk,length(data.event.(fields{ii})(jj).sample)); end;
            
            if kk == 1 || (length(cfg.(fields{ii}).triallen) > 1 && ...
                    cfg.(fields{ii}).triallen(kk) ~= cfg.(fields{ii}).triallen(kk-1))
                % Make/update indexes for time windows over which features will be computed wrt
                % beginning of trial, don't recompute if triallen was the same as before
                ix = zeros(cfg.(fields{ii}).timewindow, ...
                    1+floor((cfg.(fields{ii}).triallen(kk)-cfg.(fields{ii}).timewindow+1)/cfg.(fields{ii}).stepsize));
                ix(1,:) = 1:cfg.(fields{ii}).stepsize:cfg.(fields{ii}).triallen(kk)-cfg.(fields{ii}).timewindow+1;
                for mm = 2:cfg.(fields{ii}).timewindow
                    ix(mm,:) = ix(mm-1,:)+1;
                end
            end
            ix_off = ix + data.event.(fields{ii})(jj).sample(kk);	% ix including sample offset
            
            % Single variable features
            for mm = 1:size(ix_off,2)
                feat_count = 0;
                for nn = 1:length(cfg.(fields{ii}).features)
                    
                    % Compute features
                    switch cfg.(fields{ii}).features{nn}
                        case 'mean'
                            data_feat(t_count+mm,feat_count+1:feat_count+size(data.(fields{ii}){jj},2)) = ...
                                mean(data.(fields{ii}){jj}(ix_off(:,mm),:),1);
                        case 'var'
                            data_feat(t_count+mm,feat_count+1:feat_count+size(data.(fields{ii}){jj},2)) = ...
                                var(data.(fields{ii}){jj}(ix_off(:,mm),:),1);
                        case 'pwr'
                            data_feat(t_count+mm,feat_count+1:feat_count+size(data.(fields{ii}){jj},2)) = ...
                                sum(data.(fields{ii}){jj}(ix_off(:,mm),:).^2,1)/size(ix_off,1);
                        case 'skew'
                            data_feat(t_count+mm,feat_count+1:feat_count+size(data.(fields{ii}){jj},2)) = ...
                                skewness(data.(fields{ii}){jj}(ix_off(:,mm),:),0,1);
                        case 'kurt'
                            data_feat(t_count+mm,feat_count+1:feat_count+size(data.(fields{ii}){jj},2)) = ...
                                kurtosis(data.(fields{ii}){jj}(ix_off(:,mm),:),0,1);
                        otherwise
                            error(['Unrecognized feature: ' cfg.(fields{ii}).features{nn}]);
                    end
                    
                    % Assign feature labels
                    if kk==1
                        for pp = feat_count+1:feat_count+size(data.(fields{ii}){jj},2)
                            feat_labels{pp} = [cfg.(fields{ii}).features{nn} '_' ...
                                num2str(pp-feat_count)];
                        end
                    end
                    
                    feat_count = feat_count + size(data.(fields{ii}){jj},2);
                end
            end
            
            % Within cell / cross-component features
            for mm = 1:size(ix_off,2)
                % Set feature count to end of single-variable features
                feat_count = nfeat(1);
                
                for pp = 1:size(data.(fields{ii}){jj},2)
                    for qq = 1:size(data.(fields{ii}){jj},2)
                        if pp >= qq; continue; end;
                        if ~isempty(cfg.(fields{ii}).withinix{jj}) && ...
                                ~any(ismember(cfg.(fields{ii}).withinix{jj},[pp qq],'rows')) && ...
                                ~any(ismember(cfg.(fields{ii}).withinix{jj},[qq pp],'rows'))
                            continue;
                        end

                        % Precompute covar, coskew, and cokurt since they are all computed with CO_MOMENTS
                        if any(strcmp('covar',cfg.(fields{ii}).withincells)) || any(strcmp('coskew',cfg.(fields{ii}).withincells)) || ...
                                any(strcmp('cokurt',cfg.(fields{ii}).withincells))
                            [~,covar,coskew,cokurt] = co_moments(data.(fields{ii}){jj}(ix_off(:,mm),[pp qq]), 0);
                            covar = covar(1,2);
                            coskew = [coskew(2,1) coskew(2,3)];
                            cokurt = [cokurt(2,1) cokurt(2,3) cokurt(2,5) cokurt(2,7)];
                        end
                        
                        for nn = 1:length(cfg.(fields{ii}).withincells)
                            featincr = 1;
                            switch cfg.(fields{ii}).withincells{nn}
                                case 'covar'
                                    data_feat(t_count+mm,feat_count+1) = covar;
                                case 'coskew'
                                    data_feat(t_count+mm,feat_count+1:feat_count+2) = coskew;
                                    featincr = 2;
                                case 'cokurt'
                                    data_feat(t_count+mm,feat_count+1:feat_count+4) = cokurt;
                                    featincr = 4;
                                case 'corrcoef'
                                    C = corrcoef(data.(fields{ii}){jj}(ix_off(:,mm),[pp qq]));
                                    data_feat(t_count+mm,feat_count+1) = C(1,2);
                                case 'dbgainvar'
                                    data_feat(t_count+mm,feat_count+1) = ...
                                        10*log10(var(data.(fields{ii}){jj}(ix_off(:,mm),pp))/ ...
                                        var(data.(fields{ii}){jj}(ix_off(:,mm),qq)));
                                case 'dbgainpwr'
                                    data_feat(t_count+mm,feat_count+1) = ...
                                        10*log10(sum(data.(fields{ii}){jj}(ix_off(:,mm),pp).^2)/ ...
                                        sum(data.(fields{ii}){jj}(ix_off(:,mm),qq).^2));
                                case 'lat'
                                    S1 = std(data.(fields{ii}){jj}(ix_off(:,mm),pp));
                                    S2 = std(data.(fields{ii}){jj}(ix_off(:,mm),qq));
                                    data_feat(t_count+mm,feat_count+1) = (S1-S2)/(S1+S2);
                                otherwise
                                    error(['Unrecognized feature: ' cfg.(fields{ii}).withincells{nn}]);
                            end
                            
                            % Assign feature labels
                            if kk==1
                                feat_labels{feat_count+1} = [cfg.(fields{ii}).withincells{nn} '_cell' ...
                                    num2str(jj) '_comp' num2str(pp) '_cell' num2str(jj) '_comp' num2str(qq)];
                            end
                            
                            feat_count = feat_count + featincr;
                        end
                    end
                end
            end
            
            % Across-cell / within-component features
            for mm = 1:size(ix_off,2)
                % Set feature count to end of single-variable & within-cell features
                feat_count = nfeat(1) + nfeat(2);
                
                for pp = jj+1:length(data.(fields{ii}))
                    for qq = 1:size(data.(fields{ii}){jj},2)
                        
                        % Precompute covar, coskew, and cokurt since they are all computed with CO_MOMENTS
                        if any(strcmp('covar',cfg.(fields{ii}).withincells)) || any(strcmp('coskew',cfg.(fields{ii}).withincells)) || ...
                                any(strcmp('cokurt',cfg.(fields{ii}).withincells))
                            [~,covar,coskew,cokurt] = co_moments([data.(fields{ii}){jj}(ix_off(:,mm),qq), ...
                                data.(fields{ii}){pp}(ix_off(:,mm),qq)], 0);
                            covar = covar(1,2);
                            coskew = [coskew(2,1) coskew(2,3)];
                            cokurt = [cokurt(2,1) cokurt(2,3) cokurt(2,5) cokurt(2,7)];
                        end
                        
                        for nn = 1:length(cfg.(fields{ii}).acrosscells)
                            featincr = 1;
                            switch cfg.(fields{ii}).acrosscells{nn}
                                case 'covar'
                                    data_feat(t_count+mm,feat_count+1) = covar;
                                case 'coskew'
                                    data_feat(t_count+mm,feat_count+1:feat_count+2) = coskew;
                                    featincr = 2;
                                case 'cokurt'
                                    data_feat(t_count+mm,feat_count+1:feat_count+4) = cokurt;
                                    featincr = 4;
                                case 'corrcoef'
                                    C = corrcoef(data.(fields{ii}){jj}(ix_off(:,mm),qq), ...
                                        data.(fields{ii}){pp}(ix_off(:,mm),qq));
                                    data_feat(t_count+mm,feat_count+1) = C(1,2);
                                case 'dbgainvar'
                                    data_feat(t_count+mm,feat_count+1) = ...
                                        10*log10(var(data.(fields{ii}){jj}(ix_off(:,mm),qq))/ ...
                                        var(data.(fields{ii}){pp}(ix_off(:,mm),qq)));
                                case 'dbgainpwr'
                                    data_feat(t_count+mm,feat_count+1) = ...
                                        10*log10(sum(data.(fields{ii}){jj}(ix_off(:,mm),qq).^2)/ ...
                                        sum(data.(fields{ii}){pp}(ix_off(:,mm),qq).^2));
                                case 'lat'
                                    S1 = std(data.(fields{ii}){jj}(ix_off(:,mm),qq));
                                    S2 = std(data.(fields{ii}){pp}(ix_off(:,mm),qq));
                                    data_feat(t_count+mm,feat_count+1) = (S1-S2)/(S1+S2);
                                otherwise
                                    error(['Unrecognized feature: ' cfg.(fields{ii}).acrosscells{nn}]);
                            end
                            
                            % Assign feature labels
                            if kk==1
                                feat_labels{feat_count+1} = [cfg.(fields{ii}).withincells{nn} '_cell' ...
                                    num2str(jj) '_comp' num2str(qq) '_cell' num2str(pp) '_comp' num2str(qq)];
                            end
                            
                            feat_count = feat_count + featincr;
                        end
                    end
                end
                
            end
            
            % Set time dimension labels to the event value (for classification purposes)
            time_labels(t_count+1:t_count+size(ix_off,2)) = data.event.(fields{ii})(jj).value(kk);
            t_count = t_count + size(ix_off,2);
        end
        
        data_out.(fields{ii}){jj} = data_feat;
        data_out.dim.time.(fields{ii}){jj} = time_labels;
        data_out.dim.feat.(fields{ii}){jj} = feat_labels;
    end
    
    data_out.dim.(fields{ii}) = 'time_feat';
    data_out.fsample.(fields{ii}) = cfg.(fields{ii}).stepsize/data.fsample.(fields{ii});
end
data_out.event = [];

data_out.cfg = data.cfg;
data_out = co_logcfg(cfg,data_out);
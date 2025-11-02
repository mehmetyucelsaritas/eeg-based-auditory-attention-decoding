function [feat_x, feat_y, onehotmap] = co_data2class(cfg, data)
% CO_DATA2CLASS converts data to classification features and labels.
%
% INPUTS:
% * Note: There should only be one cfg.FIELD
% cfg.FIELD.dim     = ['time'], dimension containing classification labels (i.e. not the feature
%                     dimensions!)
% cfg.FIELD.cells   = [1], cells containing features. All cells must have the same dim length.
% cfg.FIELD.onehot  = ['no']/'yes' whether to use one-hot encoding. Classification labels must be
%                     either numbers or strings, but not both.
%
% OUTPUTS:
% feat_x            = array of features with dimensions trial x feature.
% feat_y            = cell array of classification labels if not using one-hot enclding. If one-hot
%                     encoding is enabled, this will have dimensions trial x one-hot.
% onehotmap         = cell array mapping classification labels to one-hot encoding index.
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZ
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);

assert(length(fields)==1,'There should only be one cfg.FIELD.');
if ~isfield(cfg.(fields{1}),'dim'); cfg.(fields{1}).dim = 'time'; end;
if ~isfield(cfg.(fields{1}),'cells'); cfg.(fields{1}).cells = 1; end;
if ~isfield(cfg.(fields{1}),'onehot'); cfg.(fields{1}).onehot = 'no'; end;

% Place cfg.FIELD.dim first
dimix = find(strcmp(cfg.(fields{1}).dim,dim.(fields{1})));
cfgtmp = [];
cfgtmp.(fields{1}).shift = dimix-1;
data = co_shiftdim(cfgtmp,data);

% Compile features and class labels
feat_x = [];
if nargout > 1
    if ~isfield(data.dim,cfg.(fields{1}).dim) || ...
            ~isfield(data.dim.(cfg.(fields{1}).dim),fields{1})
        error(['No features available in data.dim.' cfg.(fields{1}).dim '.' fields{1}]);
    end
    feat_y = data.dim.(cfg.(fields{1}).dim).(fields{1}){1}(:);
end

for ii = cfg.(fields{1}).cells
    sz = size(data.(fields{1}){ii});
    feat_x = [feat_x reshape(data.(fields{1}){ii},sz(1),prod(sz(2:end)))];
    if ii >= 2 && nargout > 1
        assert(isfield(data.dim,cfg.(fields{1}).dim) && ...
            isfield(data.dim.(cfg.(fields{1}).dim),fields{1}) && ...
            length(data.dim.(cfg.(fields{1}).dim).(fields{1}){ii}) == length(feat_y), ...
            ['Class label mismatch between cell 1 and cell ' num2str(ii) '.']);
        for jj = 1:length(feat_y)
            assert(all(data.dim.(cfg.(fields{1}).dim).(fields{1}){ii}{jj} == feat_y{jj}), ...
                ['Class label mismatch between cell 1 and cell ' num2str(ii) '.']);
        end
    end
end

% Convert to one-hot encoding
if strcmp(cfg.(fields{1}).onehot,'yes')
    % Error out if feat_y cell array contains mixed data types
    cellclass = class(feat_y{1});
    ciscellclass = cellfun('isclass',feat_y,cellclass);
    assert(all(ciscellclass(:)), 'One-hot encoding cannot handle mixed classification label data types');
    
    if isnumeric(feat_y{1}); feat_y = cell2mat(feat_y); end;
    [onehotmap,~,ix] = unique(feat_y);
    feat_y = zeros(length(feat_y),length(onehotmap));
    for ii = 1:size(feat_y,1)
        feat_y(ii,ix(ii)) = 1;
    end
else
    onehotmap = [];
end
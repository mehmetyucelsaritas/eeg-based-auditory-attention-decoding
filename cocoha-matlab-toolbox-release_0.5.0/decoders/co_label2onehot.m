function [vec,labels] = co_label2onehot(cfg,data)
% CO_LABEL2ONEHOT converts data labels along a specified dimension to one-hot encoding for
% classification purposes (e.g. a softmax layer in a neural network).
%
% INPUTS:
% cfg.FIELD.dim     = ['time'] dimension to convert.
% cfg.FIELD.cell    = [1] cell to convert.
% cfg.FIELD.labels  = {} labels in one-hot encoding order, specified as a cell array.
% data              = data with specified dimension to be encoded. Labels not found in the specified
%                     cell array will be returned as a row of zeros in the output. If empty, labels
%                     will be found in the dimension labels.
%
% OUTPUTS:
% vec               = one-hot encoded vector.
% labels            = unique labels, arranged in order of one-hot encoding.
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZ
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);

for ii = 1:length(fields)
    if ~isfield(cfg.(fields{ii}),'dim'); cfg.(fields{ii}).dim = 'time'; end;
    if ~isfield(cfg.(fields{ii}),'cell'); cfg.(fields{ii}).cell = 1; end;
    if ~isfield(cfg.(fields{ii}),'labels'); cfg.(fields{ii}).labels = {}; end;
    
    assert(isfield(data.dim,cfg.(fields{ii}).dim) && ...
        isfield(data.dim.(cfg.(fields{ii}).dim),fields{ii}), ...
        'No dimension information available for specified dimension.');
    assert(length(data.dim.(cfg.(fields{ii}).dim).(fields{ii})) >= cfg.(fields{ii}).cell, ...
        'No dimension information avaialble for specified cell.');
    
    if isempty(cfg.(fields{ii}).labels)
        labels = data.dim.(cfg.(fields{ii}).dim).(fields{ii}){cfg.(fields{ii}).cell};
        if iscell(labels) && ~ischar(labels{1})
            labels = cell2mat(labels);
        end
        [labels,~,ix] = unique(labels);
        vec = zeros(length(data.dim.(cfg.(fields{ii}).dim).(fields{ii}){cfg.(fields{ii}).cell}),length(labels));
        vec(sub2ind(size(vec),1:size(vec,1),ix(:)')) = 1;
    else
        labels = cfg.(fields{ii}).labels;
        vec = zeros(length(data.dim.(cfg.(fields{ii}).dim).(fields{ii}){cfg.(fields{ii}).cell}), ...
            length(labels));
        if ischar(data.dim.(cfg.(fields{ii}).dim).(fields{ii}){cfg.(fields{ii}).cell}{1})
            for jj = 1:size(vec,1)
                vec(jj,:) = strcmp( ...
                    data.dim.(cfg.(fields{ii}).dim).(fields{ii}){cfg.(fields{ii}).cell}{jj},labels);
            end
        else
            labels_mat = cell2mat(labels);
            for jj = 1:size(vec,1)
                ix = find(labels_mat == ...
                    data.dim.(cfg.(fields{ii}).dim).(fields{ii}){cfg.(fields{ii}).cell}{jj});
                if ~isempty(ix); vec(jj,ix(1)) = 1; end;
            end
        end
    end
end
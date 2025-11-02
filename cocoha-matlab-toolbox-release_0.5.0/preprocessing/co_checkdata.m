function dim = co_checkdata(cfg,data,varargin)
% CO_CHECKDATA makes sure the data structure looks ok. Only examines the fields present in cfg.
%
% INPUTS:
% cfg       = cfg structure with fields that should also exist in data as a data source (e.g. eeg, 
%             wav). Only these fields will be checked.
% data      = data structure to be checked.
% varargin  = list of fields in cfg to be ignored (apart from 'date' and 'fcn'). 
%
% OUTPUTS:
% dim       = contains fields corresponding to data fields. Each field is a cell array with the 
%             dimension labels.
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU/Oticon, UCL, UZH
% Author(s): Daniel D.E. Wong

dim = [];
if ~isstruct(cfg); return; end;
fields = fieldnames(cfg);
for ii = 1:length(fields)
    if strcmp(fields{ii},'date') || strcmp(fields{ii},'fcn'); continue; end;    % Fields used for logging
    if ~isempty(varargin) && any(strcmp(fields{ii},varargin)); continue; end;   % Explicitly excluded fields
    
    assert(isfield(data,fields{ii}), ['Field ' fields{ii} ' does not exist in data.']);
    assert(isfield(data,'fsample') & isfield(data.fsample,fields{ii}), ...
        ['Sampling rate for ' fields{ii} ' is missing in data.fsample.']);
    assert(isfield(data,'dim') & isfield(data.dim,fields{ii}), ...
        ['Dimension information for ' fields{ii} ' is missing in data.dim.']);

    % Check optional dimension axis labels
    dim.(fields{ii}) = co_strsplit(data.dim.(fields{ii}),'_');
    for jj = 1:length(dim.(fields{ii}))
        if isfield(data.dim,dim.(fields{ii}){jj}) && ...                        % dimension ix
                isfield(data.dim.(dim.(fields{ii}){jj}),fields{ii})
            assert(length(data.dim.(dim.(fields{ii}){jj}).(fields{ii})) == ...
                length(data.(fields{ii})), ...
                ['The number of cells in the dimension labels for dimension ' ...
                dim.(fields{ii}){jj} ' does not match cells in data.' fields{ii} '.']);
            for kk = 1:length(data.(fields{ii}))                                % field cell
                assert(length(data.dim.(dim.(fields{ii}){jj}).(fields{ii}){kk}) == ...
                    size(data.(fields{ii}){kk},jj), ...
                    ['The length of dimension ' dim.(fields{ii}){jj} ' does not match data.' ... 
                    fields{ii} '{' num2str(kk) '}.']);
            end
        end
    end
end
assert(isfield(data,'cfg'),'Data structure is missing .cfg field (cell array).');
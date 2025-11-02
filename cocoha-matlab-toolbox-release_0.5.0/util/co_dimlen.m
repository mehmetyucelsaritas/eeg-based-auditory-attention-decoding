function len = co_dimlen(cfg,data)
% CO_DIMLEN returns the size of the data along a given dimension, for each cell.
%
% INPUTS:
% Note that Only one data FIELD can be specified at a time in cfg.
% cfg.FIELD.dim     = dimension name (e.g. 'time').
% cfg.FIELD.cell    = ['all'] / index of cells for which to return the dimension length.
%
% data
%
%
% OUTPUTS:
% len               = array of dimension lengths for the specified dimension, for each specified
%                     cell.
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZH
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);

assert(length(fields)==1,'Only one cfg.FIELD supported as input to CO_DIMLEN.');

% Defaults
if ~isfield(cfg.(fields{1}),'cell'); cfg.(fields{1}).cell = 'all'; end;

if ischar(cfg.(fields{1}).cell) && strcmp(cfg.(fields{1}).cell,'all')
    cellix = 1:length(data.(fields{1}));
end

len = zeros(1,length(cellix));

dimix = find(strcmp(cfg.(fields{1}).dim,dim.(fields{1})));

for ii = 1:length(len)
    len(ii) = size(data.(fields{1}){ii},dimix);
end
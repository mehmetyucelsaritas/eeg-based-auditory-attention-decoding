function sz = co_size(cfg,data)
% CO_DATASIZE returns the size of all dimensions a particular trial in a data field as an array.
% This is different from MATLAB's SIZE function as the number of elements in the returned array will
% match the number of dimensions in the data, as indicated in data.dim.FIELD, even if one of the
% dimensions is a singleton dimension.
%
% INPUTS:
% cfg.FIELD.cell    = [1] the trial/cell for which to return the size.
%
% OUTPUTS:
% sz                = array of dimension sizes.
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZ
% Author(s): Daniel D.E. Wong


dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);

assert(length(fields)==1,'Only one data field should be specified in cfg.');
if ~isfield(cfg.(fields{1}),'cell'); cfg.(fields{1}).cell = 1; end;

sz = size(data.(fields{1}){cfg.(fields{1}).cell});
singletons = length(dim.(fields{1}))-length(sz);    % Number of trailing singleton dimensions
if singletons < 0; singletons = 0; end;             % In case the column is a singleton dimension but not listed as a dimension in the data structure
sz = [sz, ones(1,singletons)];
sz = sz(1:length(dim.(fields{1})));
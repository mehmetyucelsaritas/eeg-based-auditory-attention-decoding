function data = co_logcfg(cfg,data)
% CO_LOGCFG is a general sub-routine used by many COCOHA toolbox functions to log the cfg settings
% passed to them. Will not log the cfg structure if the calling function is called by another co_*
% function.
%
% INPUTS:
% cfg   = cfg parameter passed to calling function
% data  = 
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU/Oticon, UCL, ETH Zurich
% Author(s): Daniel D.E. Wong

[stck,stckix] = dbstack;
if ~(length(stck) >= stckix+2 && length(stck(stckix+2).name) >= 3 && strcmp(stck(stckix+2).name(1:3),'co_'))
    cfg.fcn = stck(stckix+1).name;
    cfg.date = date;
    if ~isfield(data,'cfg'); data.cfg{1} = cfg; else data.cfg{end+1} = cfg; end;
end
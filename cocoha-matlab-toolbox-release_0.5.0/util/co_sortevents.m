function data = co_sortevents(cfg,data)
% CO_SORTEVENTS sorts events in ascending order based on the sample time.
% 
% The cfg structure does not take any arguments and is only there for compatibility with the
% toolbox's function structure.
%
% INPUTS:
% cfg   = []. The cfg structure does not take any arguments and is only there for compatibility with
%         the toolbox's function structure.
%
% data
%
% OUTPUTS:
% data  = data structure with events arranged in temporally increasing order.
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU/Oticon, UCL, ETH Zurich
% Author(s): Daniel D.E. Wong

if isempty(data.event); data = co_logcfg(cfg,data); return; end;
evtfields = fieldnames(data.event);
for ii = 1:length(evtfields)
    for jj = 1:length(data.event.(evtfields{ii}))
        [data.event.(evtfields{ii})(jj).sample,ix] = sort(data.event.(evtfields{ii})(jj).sample);
        data.event.(evtfields{ii})(jj).value = data.event.(evtfields{ii})(jj).value(ix);
    end
end

data = co_logcfg(cfg,data);
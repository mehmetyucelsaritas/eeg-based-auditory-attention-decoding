function data = co_selectevent(cfg,data)
% CO_SELECTEVENT selects events according to the parameters specified.
%
% INPUTS:
% cfg.FIELD.cell    = ['all']/array of cell indexes to operate on.
% cfg.FIELD.event   = ['all']/'none'/cell array of event values to select/array of event indexes to
%                     select.
%
% data
%
% OUTPUTS:
% data
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZ
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);

for ii = 1:length(fields)
    if ~isfield(cfg.(fields{ii}),'cell'); cfg.(fields{ii}).cell = 'all'; end;
    if ischar(cfg.(fields{ii}).cell)
        assert(strcmp(cfg.(fields{ii}).cell,'all'),['Invalid value for cfg.' fields{ii} ,'.cell.']);
        cellix = 1:length(data.(fields{ii}));
    else
        cellix = cfg.(fields{ii}).cell;
    end
    
    for jj = cellix
        if ischar(cfg.(fields{ii}).event)
            if strcmp(cfg.(fields{ii}).event,'all')
                break;  % Nothing to do!
            elseif strcmp(cfg.(fields{ii}).event,'none')
                % Clear events
                data.event.(fields{ii})(jj).sample = [];
                data.event.(fields{ii})(jj).value = {};
            end
        elseif iscell(cfg.(fields{ii}).event)       % Cell array of event values to keep
            keepix = [];
            for kk = 1:length(cfg.(fields{ii}).event)
                for mm = 1:length(data.event.(fields{ii})(jj).value)
                    if length(cfg.(fields{ii}).event{kk}) == ...
                            length(data.event.(fields{ii})(jj).value{mm}) && ...
                            all(cfg.(fields{ii}).event{kk}==data.event.(fields{ii})(jj).value{mm})
                        keepix = [keepix mm];
                    end
                end
            end
            keepix = unique(keepix);    % Unique already sorts in ascending order
            data.event.(fields{ii})(jj).sample = data.event.(fields{ii})(jj).sample(keepix);
            data.event.(fields{ii})(jj).value = data.event.(fields{ii})(jj).value(keepix);
        elseif isnumeric(cfg.(fields{ii}).event)    % Array of event indexes to keep
            keepix = unique(cfg.(fields{ii}).event(:));
            data.event.(fields{ii})(jj).sample = data.event.(fields{ii})(jj).sample(keepix);
            data.event.(fields{ii})(jj).value = data.event.(fields{ii})(jj).value(keepix);
        end
    end
end

% Save cfg settings for future reference
data = co_logcfg(cfg,data);
function data = co_aligndata(cfg, data)
% CO_ALIGNDATA aligns one data field (e.g. audio) with a reference field (e.g. EEG). Both fields
% must have a dimension in common. Note that event times for the shifted field will also be altered
% if the data is realigned in the time dimension. The shifted field will have the same number of 
% time samples as the reference field.
%
% INPUTS:
% cfg.dim               = ['time'], name of dimension to be realigigned
% cfg.FIELD.type        = 'ref'/['aln'] whether this field is used as a size reference or will be 
%                         aligned
% cfg.FIELD.cell        = [1] for cfg.FIELD.type == 'ref'
%                         ['all'] for cfg.FIELD.type == 'aln'
%                         cell number containing data being used as a reference or being aligned. 
%                         Can be set to 'all' if cfg.FIELD.type is 'aln', but must be a single
%                         number for cfg.FIELD.type == 'ref'.
% cfg.FIELD.alnsample   = the sample number of the ref data to which the aln data will be shifted.
%
% data                  = data strucure.
%
%
% OUTPUTS:
% data                  = data structure with alignd fields.
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU/Oticon, UCL, ETH Zurich
% Author(s): Daniel D.E. Wong


% Default to time dimension
if ~isfield(cfg,'dim'); cfg.dim = 'time'; end;

dim = co_checkdata(cfg,data,'dim');

% Find reference field and set defaults
fields = fieldnames(rmfield(cfg,'dim'));
refix = 0;
for ii = 1:length(fields)
    if ~isfield(cfg.(fields{ii}),'type')
        cfg.(fields{ii}).type = 'aln';
        if ~isfield(cfg.(fields{ii}),'cell');
            cfg.(fields{ii}).cell = 'all';
        end
    elseif strcmp(cfg.(fields{ii}).type,'ref')
        if ~refix; refix = ii; else error('More than one reference field specified.'); end;
        if ~isfield(cfg.(fields{ii}),'cell');
            cfg.(fields{ii}).cell = 1;
        end
    end
end
if ~refix; warning('No reference field specified. Align fields will be referenced against themselves.'); end;

for ii = 1:length(fields)
    if ii == refix; continue; end;
    if ~refix; refix2 = ii; else refix2 = refix; end;   % If ref not specified (refix2 is working ref index)
    assert(~strcmp(cfg.dim,'time') | data.fsample.(fields{ii}) == data.fsample.(fields{refix}), ...
        ['Sampling rate for ' fields{ii} ' does not match the reference (' fields{refix} ').']);
    
    dimixref = strmatch(cfg.dim,dim.(fields{refix2}),'exact');
    dimix = strmatch(cfg.dim,dim.(fields{ii}),'exact');
    
    % Set list of cells to work on
    if isstr(cfg.(fields{ii}).cell) & strcmp(cfg.(fields{ii}).cell,'all')
        cfg.fields{ii}.cell = 1:length(data.(fields{ii}));
    end
    
    alnsample = cfg.(fields{ii}).alnsample;
    
    for jj = cfg.fields{ii}.cell
        % Set default cell index in reference data to use
        if refix2==ii
            refcell = jj;
        else
            if ~isfield(cfg.(fields{refix2}),'cell'); cfg.(fields{refix2}).cell = 1; end;
            refcell = cfg.(fields{refix2}).cell;
        end
        
        datatmp = data.(fields{ii}){jj};    % Holds data to be copied
        if dimix > 1                        % Rearrange dimensions so dimdim is 1
            datatmp = shiftdim(datatmp, dimix-1);
        end
        sztmp = size(datatmp);
        
        % Create array to be pasted into (first dimension sized to ref data) 
        data.(fields{ii}){jj} = zeros([size(data.(fields{refix2}){refcell},dimixref) sztmp(2:end)]);
        sz = size(data.(fields{ii}){jj});
        
        if alnsample > 1 && alnsample <= sz(1)
            data.(fields{ii}){jj}(alnsample:min(sz(1),alnsample+sztmp(1)-1),:) = ...
                datatmp(1:min(sztmp(1),sz(1)-alnsample+1),:);
        elseif alnsample < 1 && alnsample+sztmp(1)>1
            data.(fields{ii}){jj}(1:min(sz(1),alnsample+sztmp(1)-1),:) = ...
                datatmp(-alnsample+2:min(alnsample+sz(1)-1,sztmp(1)),:);
        end
        
        if dimix > 1   % Rearrange dimensions to original order
            data.(fields{ii}){jj} = shiftdim(data.(fields{ii}){jj}, length(dim.(fields{ii}))-dimix+1);
        end
        
        % Shift event sample info
        if strcmp(cfg.dim,'time') && isfield(data.event,fields{ii})
            data.event.(fields{ii})(cfg.fields{ii}.cell).sample = data.event.(fields{ii})(cfg.fields{ii}.cell).sample + alnsample;
        end
        
        % Set dim axis labels to those of ref data
        if isfield(data.dim, cfg.dim) && isfield(data.dim.(cfg.dim),fields{refix2})
            data.dim.(cfg.dim).(fields{ii}){jj} = data.dim.(cfg.dim).(fields{refix2}){refcell};
        end
    end
end

% Save cfg settings for future reference
data = co_logcfg(cfg,data);
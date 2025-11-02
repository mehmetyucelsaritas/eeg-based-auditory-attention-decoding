function [data] = co_splitdata(cfg,data)
% CO_SPLITDATA splits a data field at specified samples into additional cells. Newly created cells
% will be positioned after the cell from which they were created.
%
% INPUTS
% cfg.FIELD.dim             = ['time'] name of dimension for splitting data
% cfg.FIELD.splitsample     = sample(s) where to split data field.
% cfg.FIELD.splitcell       = ['all'], cells to be split.
% data                      = 
%
%
% OUTPUTS:
% data                      = 
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU/Oticon, UCL, UZH
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);
for ii = 1:length(fields)
    if ~isfield(cfg.(fields{ii}),'dim'); cfg.(fields{ii}).dim = 'time'; end;
    if ~isfield(cfg.(fields{ii}),'splitcell'); cfg.(fields{ii}).splitcell = 'all'; end;
    %assert(~strcmp(cfg.(fields{ii}).dim,'chan'), 'CO_SPLITDATA was not designed for splitting channels.');
    
    dimname = cfg.(fields{ii}).dim;
    dimix = find(strcmp(dimname,dim.(fields{ii})));
    assert(~isempty(dimix), ['Dimension ' dimname ' does not exist.']);
    
    if dimix > 1
        cfgtmp = []; cfgtmp.(fields{ii}).shift = dimix-1; data = co_shiftdim(cfgtmp,data);
    end
    
    splitsample = cfg.(fields{ii}).splitsample(:)';
    splitsample = sort(splitsample,'descend');
    if isstr(cfg.(fields{ii}).splitcell) & strcmp(cfg.(fields{ii}).splitcell,'all')
        splitcell = 1:length(data.(fields{ii}));
    else
        splitcell = cfg.(fields{ii}).splitfields;
    end
    
    % Predicted offsets of existing cells after splitting
    splitoffsets = splitcell + [0:length(splitcell)-1]*length(splitsample);
    
    for jj = splitoffsets
%         if dimix > 1   % Rearrange dimensions so dimdim is 1
%             data.(fields{ii}){jj} = shiftdim(data.(fields{ii}){jj}, dimix-1);
%         end
        
        for ss = 1:length(splitsample)
            if splitsample(ss) == 1; continue; end; % Ignore splits at first index
            assert(splitsample(ss) < size(data.(fields{ii}){jj},1), ...
                ['cfg.' fields{ii} '.splitsample(' num2str(ss) ') too large.']);

            sz = size(data.(fields{ii}){jj});
            data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},sz(1),prod(sz(2:end)));

            tmpfield = data.(fields{ii})(jj+1:end);     % Move subsequent fields to the end
            data.(fields{ii}){jj+1} = data.(fields{ii}){jj}(splitsample(ss):end,:);
            data.(fields{ii}){jj+1} = reshape(data.(fields{ii}){jj+1},[sz(1)-splitsample(ss)+1 sz(2:end)]);
            data.(fields{ii}){jj} = data.(fields{ii}){jj}(1:splitsample(ss)-1,:);
            data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj},[splitsample(ss)-1 sz(2:end)]);
            data.(fields{ii})(jj+2:jj+1+length(tmpfield)) = tmpfield;

%             if dimix > 1
%                 data.(fields{ii}){jj+1} = shiftdim(data.(fields{ii}){jj+1}, length(dim.(fields{ii}))-dimix+1);
%             end
            
            % Split dimension data if applicable
            if isfield(data.dim,dimname) && isfield(data.dim.(dimname),fields{ii}) && ...
                    ~isempty(data.dim.(dimname).(fields{ii}){jj})
                tmpdim = data.dim.(dimname).(fields{ii})(jj+1:end);  % Move subsequent dimension data to end
                data.dim.(dimname).(fields{ii}){jj+1} = data.dim.(dimname).(fields{ii}){jj}(splitsample(ss):end);
                data.dim.(dimname).(fields{ii}){jj} = data.dim.(dimname).(fields{ii}){jj}(1:splitsample(ss)-1);
                data.dim.(dimname).(fields{ii})(jj+2:jj+1+length(tmpdim)) = tmpdim;
            end
            
            % Copy other dimension data to newly created cells
            for kk = 1:length(dim.(fields{ii}))
                dimnamekk = dim.(fields{ii}){kk};
                if strcmp(dimnamekk,dimname); continue; end;     % We just took care of this
                
                if isfield(data.dim,dimnamekk) && isfield(data.dim.(dimnamekk),fields{ii}) && ...
                        ~isempty(data.dim.(dimnamekk).(fields{ii}){jj})
                    tmpdim = data.dim.(dimnamekk).(fields{ii})(jj+1:end);
                    data.dim.(dimnamekk).(fields{ii})(jj+1) = data.dim.(dimnamekk).(fields{ii})(jj);
                    data.dim.(dimnamekk).(fields{ii})(jj+2:jj+1+length(tmpdim)) = tmpdim;
                end
            end
            
            % Move event samples
            if isfield(data.event,fields{ii})
                tmpevt = data.event.(fields{ii})(jj+1:end);
                if strcmp(dimname,'time')
                    evtix = find(data.event.(fields{ii})(jj).sample >= splitsample(ss));
                    data.event.(fields{ii})(jj+1).sample = data.event.(fields{ii})(jj).sample(evtix)-splitsample(ss)+1;
                    data.event.(fields{ii})(jj+1).value = data.event.(fields{ii})(jj).value(evtix);
                    data.event.(fields{ii})(jj).sample(evtix) = [];
                    data.event.(fields{ii})(jj).value(evtix) = [];
                else
                    data.event.(fields{ii})(jj+1) = data.event.(fields{ii})(jj);
                end
                data.event.(fields{ii})(jj+2:jj+1+length(tmpevt)) = tmpevt;
            end
        end
%         if dimix > 1   % Rearrange dimensions to original order
%             data.(fields{ii}){jj} = shiftdim(data.(fields{ii}){jj}, length(dim.(fields{ii}))-dimix+1);
%         end
    end
    if dimix > 1
        cfgtmp = []; cfgtmp.(fields{ii}).shift = 1-dimix; data = co_shiftdim(cfgtmp,data);
    end
end

% Save cfg settings for future reference
data = co_logcfg(cfg,data);
function data = co_mfcc(cfg, data)
% CO_MFCC transforms audio data into MFCC coefficients.
%
% INPUTS
% cfg.FIELD.frameshift      = frame shift (ms).
% cfg.FIELD.frameduration   = [25] frame duration (ms).
% cfg.FIELD.alpha           = [0.97] preemphasis coefficient.
% cfg.FIELD.window          = [@hamming] windowing function handle.
% cfg.FIELD.frange          = [300 3700] frequency range to consider.
% cfg.FIELD.fchan           = [30] number of frequency bank channels.
% cfg.FIELD.ncoeff          = [13] number of cepstral coefficients.
% cfg.FIELD.lift            = [22] cepstral sine lifter parameter.
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZ
% Author(s): Daniel D.E. Wong

if ~exist('mfcc','file')
    addpath(fullfile(fileparts(which('co_defaults')), 'external', 'mfcc'));
end

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);

for ii = 1:length(fields)
    if ~isfield(cfg.(fields{ii}),'frameduration'); cfg.(fields{ii}).frameduration = 25; end;
    if ~isfield(cfg.(fields{ii}),'alpha'); cfg.(fields{ii}).alpha = 0.97; end;
    if ~isfield(cfg.(fields{ii}),'window'); cfg.(fields{ii}).window = @hamming; end;
    if ~isfield(cfg.(fields{ii}),'frange'); cfg.(fields{ii}).frange = [300 3700]; end;
    if ~isfield(cfg.(fields{ii}),'fchan'); cfg.(fields{ii}).fchan = 30; end;
    if ~isfield(cfg.(fields{ii}),'ncoeff'); cfg.(fields{ii}).ncoeff = 13; end;
    if ~isfield(cfg.(fields{ii}),'lift'); cfg.(fields{ii}).lift = 22; end;
    
    % Check that input is time x chan
    assert(length(dim.(fields{ii}))==2 && any(strcmp(dim.(fields{ii}),'time')) && ...
        any(strcmp(dim.(fields{ii}),'chan')), ...
        ['data.', fields{ii}, 'should have dimensions time_chan.']);
    if strcmp(dim.(fields{ii}){2},'time')
        cfgtmp = []; cfgtmp.(fields{ii}).shift = 1;
        data = co_shiftdim(cfgtmp,data);
    end
    
    fs = data.fsample.(fields{ii});
    
    for jj = 1:length(data.(fields{ii}))
        cc_all = [];
        for kk = 1:size(data.(fields{ii}),2)
            cc = mfcc(data.(fields{ii}){jj}, fs, ...
                cfg.(fields{ii}).frameduration, cfg.(fields{ii}).frameshift, ...
                cfg.(fields{ii}).alpha, cfg.(fields{ii}).window, cfg.(fields{ii}).frange, ...
                cfg.(fields{ii}).fchan, cfg.(fields{ii}).ncoeff, cfg.(fields{ii}).lift);
            if isempty(cc_all); cc_all = zeros(cat(2,size(cc'),size(data.(fields{ii}),2))); end;
            cc_all(:,:,kk) = cc';
        end
        data.(fields{ii}){jj} = cc_all;
    end
    
    % Recalculate sampling frequency
    data.fsample.(fields{ii}) = 1/cfg.(fields{ii}).frameshift*1000;
    
    % Adjust event samples
    if isfield(data.event,fields{ii})
        for jj = 1:length(data.event.(fields{ii}))
            data.event.(fields{ii})(jj).sample = data.event.(fields{ii})(jj).sample * ...
                data.fsample.(fields{ii})/fs;
        end
    end
    
    % Adjust dimension descriptor
    data.dim.(fields{ii}) = 'time_mfcc_chan';
end
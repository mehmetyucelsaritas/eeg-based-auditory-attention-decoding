function data = co_decode_regression(cfg,data,decoder)
% CO_DECODE_REGRESSION uses a trained regression decoder to decode the specified input (in the case
% of the reverse correlation algorithm), or to jointly decode the specified input and 'output' (in
% the case of the CCA algorithm). The input (and output in the case of CCA) must contain the same
% non-time (/feature) dimensions as the data used to train the decoder.
%
% INPUTS
% cfg.input.field           = field name of the input data.
% cfg.input.cell            = ['all']/array of cell number(s) of the input data.
% cfg.output.field          = field name of the output data. For a CCA decoder, this field must
%                             exist in the data and have cells organized in the same way as the
%                             input.
%
% cfg.cca.beamforming       = {'no','no'} whether to perform beamforming on input and/or output
%                             fields when the CCA decoder is used.
% cfg.cca.beamsup           = [] indexes of CCA components to use for beamformer suppression.
% cfg.cca.beamsupext        = {[],[]} additional forward weights to suppress when beamforming is
%                             used.
% cfg.cca.beamreg           = {[],[]} beamforming covariance regularization matrix.
%
% data
% decoder                   = a CoDecoder object obtained from CO_TRAIN_REGRESSION.
%
% OUTPUTS
% data                      = data structure containing decoder output with the same field name as
%                             the target output used in the training data.
%
% See also: CO_TRAIN_REGRESSION
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU/Oticon, UCL, UZH
% Author(s): Daniel D.E. Wong


if ~exist('FindTRF','file')
    disp('Adding the Telluride Decoding Toolbox to the MATLAB path.');
    addpath(fullfile(fileparts(which('co_defaults')), 'external', 'telluride-decoding-toolbox'));
end
if ~exist('nt_relshift','var')
    addpath(fullfile(fileparts(which('co_defaults')), 'external', 'NoiseTools'));
    addpath(fullfile(fileparts(which('co_defaults')), 'external', 'NoiseTools', 'COMPAT'));
end

assert(strcmp(decoder.type,'regression'), 'Decoder is not of type ''regression''.');

cfgtmp = [];
cfgtmp.(cfg.input.field) = [];
if isfield(data,cfg.output.field); cfgtmp.(cfg.output.field) = []; end;
dim = co_checkdata(cfgtmp,data);

% Set defaults
if ~isfield(cfg.input,'cell'); cfg.input.cell = 'all'; end;

% Determine which input cells to process
if ischar(cfg.input.cell) && strcmp(cfg.input.cell,'all')
    incell = 1:length(data.(cfg.input.field));
else
    incell = cfg.input.cell;
end

% Check sampling rate (for reverse correlation decoder)
if isfield(decoder.spec,'fsample')
    assert(decoder.spec.fsample==data.fsample.(cfg.input.field), ...
        'Input sampling rate does not match that of decoder.');
end


% Make decoder.dim the first dimension in the input data
indimix = find(strcmp(decoder.dim.dim,dim.(cfg.input.field)));
assert(~isempty(indimix),['Dimension ''' decoder.dim.dim ''' is missing from input.']);
cfgtmp = [];
cfgtmp.(cfg.input.field).shift = indimix-1;
data = co_shiftdim(cfgtmp,data);

% Remove output dimension axis values if they exist
if isfield(data,cfg.output.field)
    for ii = 1:length(dim.(cfg.output.field))
        if isfield(data.dim,dim.(cfg.output.field){ii}) && ...
                isfield(data.dim.(dim.(cfg.output.field){ii}),cfg.output.field)
            data.dim.(dim.(cfg.output.field){ii}) = ...
                rmfield(data.dim.(dim.(cfg.output.field){ii}),cfg.output.field);
            if isempty(data.dim.(dim.(cfg.output.field){ii})); data.dim = rmfield(data.dim,dim.(cfg.output.field){ii}); end;
        end
    end
end

% Data structure-level processing
switch decoder.spec.regressalgo
    case 'reversecorr'
        % Set output sampling rate from input sampling rate
        data.fsample.(cfg.output.field) = data.fsample.(cfg.input.field);
        
        % Set output dimensions
        data.dim.(cfg.output.field) = decoder.dim.outdim;
    case 'cca'
        % CCA specific defaults
        if ~isfield(cfg,'cca') || ~isfield(cfg.cca,'beamforming')
            cfg.cca.beamforming = {'no','no'};
        end
        if ~isfield(cfg.cca,'beamsup'); cfg.cca.beamsup = []; end;
        if ~isfield(cfg.cca,'beamsupext'); cfg.cca.beamsupext = {[],[]}; end;
        if ~isfield(cfg.cca,'beamreg'); cfg.cca.beamreg = {[],[]}; end;
        
        % Make sure input and output sampling rates match
        assert(data.fsample.(cfg.input.field) == data.fsample.(cfg.output.field), ...
            'Input and output sampling rates do not match.');
        
        % Make decoder.dim the first dimension in 'output'
        outdimix = find(strcmp(decoder.dim.dim,dim.(cfg.output.field)));
        cfgtmp = [];
        cfgtmp.(cfg.output.field).shift = outdimix-1;
        data = co_shiftdim(cfgtmp,data);

        % Remove input dimension axis values since input field will be modified
        for ii = 1:length(dim.(cfg.input.field))
            if isfield(data.dim,dim.(cfg.input.field){ii}) && ...
                    isfield(data.dim.(dim.(cfg.input.field){ii}),cfg.input.field)
                data.dim.(dim.(cfg.input.field){ii}) = ...
                    rmfield(data.dim.(dim.(cfg.input.field){ii}),cfg.input.field);
                if isempty(data.dim.(dim.(cfg.input.field){ii})); data.dim = rmfield(data.dim,dim.(cfg.input.field){ii}); end;
            end
        end

        % Set new input and output dimensions
        data.dim.(cfg.input.field) = [decoder.dim.dim '_cca'];
        data.dim.(cfg.output.field) = [decoder.dim.dim '_cca'];
    otherwise
        error('Unsupported regression algorithm.');
end

% Cell-level processing
for ii = 1:length(incell)
    indata = data.(cfg.input.field){incell(ii)};
    
    % Perform input dimension check
    assert(ndims(indata)==length(decoder.dim.indimsz),'Number of input dimensions differs from training data.');
    indatasz = size(indata);
    assert(all(indatasz(2:end) == decoder.dim.indimsz(2:end)),'Input dimension size differs from training data.');

    % Make indata 2 dimensional
    szinshift = size(indata);
    indata = reshape(indata,szinshift(1),prod(szinshift(2:end)));

    % Handle Z-scoring of input
    indata = indata-repmat(decoder.spec.meanin,size(indata,1),1);
    indata = indata./repmat(decoder.spec.stdin,size(indata,1),1);

    switch decoder.spec.regressalgo
        case 'reversecorr'
            if strcmp(decoder.spec.dir,'forward'); dir = 1; else dir = -1; end;
            [~,outdata] = FindTRF([], [], dir, indata, decoder.decoder, decoder.spec.lags, [], [], 0);
            
            % Un-Z-score output
            outdata = outdata .* repmat(decoder.spec.stdout,size(outdata,1),1);
            outdata = outdata + repmat(decoder.spec.meanout,size(outdata,1),1);
            
            % Reshape outdata
            outdimix = find(strcmp(decoder.dim.dim,co_strsplit(decoder.dim.outdim,'_')));
            outdimsz = circshift(decoder.dim.outdimsz(:),[-outdimix+1 0]);  % Dimension sizes of output AFTER decoder.dim.dim shifted to first dimension
            outdata = reshape(outdata,size(outdata,1),outdimsz(2:end));
            
            % Shift outdata to match training output dimension order
            outdata = shiftdim(outdata,mod(outdimix-1,length(decoder.dim.outdimsz)));
            data.(cfg.output.field){incell(ii)} = outdata;
            
            % Add events to output
            if isfield(data.event,cfg.output.field)
                if strcmp(decoder.dim.dim,'time')
                    data.event.(cfg.output.field)(ii).sample = data.event.(cfg.input.field)(incell(ii)).sample;
                    data.event.(cfg.output.field)(ii).value = data.event.(cfg.input.field)(incell(ii)).value;
                else
                    data.event.(cfg.output.field)(ii).sample = [];
                    data.event.(cfg.output.field)(ii).value = {};
                end
            end
        case 'cca'
            outdata = data.(cfg.output.field){incell(ii)};
            
            % Perform output dimension check
            assert(ndims(outdata)==length(decoder.dim.outdimsz), ...
                'Number of ''output'' dimensions differs from training data.');
            outdatasz = size(outdata);
            assert(all(outdatasz(2:end) == decoder.dim.outdimsz(2:end)), ...
                'Output dimension size differs from training data.');
            
            % Make 'output' two dimensional
            szoutshift = size(outdata);
            outdata = reshape(outdata,szoutshift(1),prod(szoutshift(2:end)));
            
            % Handle Z-scoring of 'output'
            outdata = outdata-repmat(decoder.spec.meanout,size(outdata,1),1);
            outdata = outdata./repmat(decoder.spec.stdout,size(outdata,1),1);
            
            A = decoder.decoder.A;
            invA = []; if isfield(decoder.decoder,'invA'); invA = decoder.decoder.invA; end;
            B = decoder.decoder.B;
            invB = []; if isfield(decoder.decoder,'invB'); invB = decoder.decoder.invB; end;
            N = min(size(A,2),size(B,2));
            
            % Beamforming
            if strcmp(cfg.cca.beamforming{1},'yes') % Input
                if isempty(invA); A = A./repmat(sqrt(sum(A.^2,1)),size(A,1),1); invA = pinv(A)'; end;
                A = beamforming(indata,invA,N,cfg.cca.beamsup,cfg.cca.beamsupext{1},cfg.cca.beamreg{1});
            end
            if strcmp(cfg.cca.beamforming{2},'yes') % Output
                if isempty(invB); B = B./repmat(sqrt(sum(B.^2,1)),size(B,1),1); invB = pinv(B)'; end;
                B = beamforming(outdata,invB,N,cfg.cca.beamsup,cfg.cca.beamsupext{2},cfg.cca.beamreg{1});
            end
            
            % Apply CCA weights to input and output
            N = min(size(A,2),size(B,2));
            indata = indata*A(:,1:N); outdata = outdata*B(:,1:N);
            [indata,outdata] = nt_relshift(indata,outdata,decoder.spec.delay);
            data.(cfg.input.field){incell(ii)} = indata;
            data.(cfg.output.field){ii} = outdata;
    end
end

% Cleanup
switch decoder.spec.regressalgo
    case 'reversecorr'
    case 'cca'
        data.(cfg.input.field) = data.(cfg.input.field)(incell);
        data.(cfg.output.field) = data.(cfg.output.field)(1:length(incell));
end

% Save cfg settings for future reference
[stck,stckix] = dbstack;
cfg.fcn = stck(stckix).name;
cfg.date = date;
cfg.datacfg = data.cfg;


function W = beamforming(x,L,N,beamsup,beamsupext,beamreg)
% Computes weight vectors using beamforming.
% x         = data.
% L         = forward mapping vectors (e.g. lead potentials).
% N         = number of lead potentials for which to compute W.
% beamsup   = indexes of lead potentials to use for suppression.
% beamsupext= extra forward mapping vectors (e.g. lead potentials) to be suppressed.

if ~isempty(beamsupext)
    beamsupext = beamsupext./repmat(sqrt(sum(beamsupext.^2,1)),size(beamsupext,1),1);
end
L = L./repmat(sqrt(sum(L.^2,1)),size(L,1),1);   % Normalize columns of L

R = x'*x; R = R/size(x,1);
if ~isempty(beamreg); R = R+beamreg; end;
invR = pinv(R);
W = zeros(size(L,1),N);
if ~isempty(beamsup)                    % Compute suppressed components together
    assert(max(beamsup)<=size(L,2) && min(beamsup)>=1, 'cfg.cca.beamsup: Beamformer suppression idexes out of bounds.');
    ll = [L(:,beamsup), beamsupext];
    invRL = invR*ll; J = inv(ll'*invRL);
    ww = invRL*J;
    bsix = find(beamsup<=N);
    W(:,beamsup(bsix)) = ww(:,bsix);
end
for jj = setdiff(1:N,beamsup)           % Compute other components with suppression
    lix = [jj, beamsup(:)'];
    ll = [L(:,lix), beamsupext];
    invRL = invR*ll; J = inv(ll'*invRL);
    ww = invRL*J;
    W(:,jj) = ww(:,1);
end
W = nt_normcol(W);  % Weight normalization
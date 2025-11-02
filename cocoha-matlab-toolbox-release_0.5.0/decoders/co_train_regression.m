function [decoder,val_log] = co_train_regression(cfg,data)
% CO_TRAIN_REGRESSION trains a linear temporal decoder filter. All other dimensions besides cfg.dim
% are treated as features; as such, the input/output dimensions are shifted so that cfg.dim is the
% first. All other dimensions are effectively reshaped into a single feature dimension. It should be
% noted that the terms 'input' and 'output' are with respect to the decoder (predicting an output
% from a given input), and not with respect to the actual system.
%
%
% INPUTS:
% cfg.dim                   = ['time'] dimension to decode on.
% cfg.input.field           = field name and input data (e.g. audio for forward model, EEG for 
%                             reverse model).
% cfg.input.cell            = [1] cell number of input data.
% cfg.output.field          = field name ouput data (e.g. eeg for forward model, audio for reverse 
%                             model).
% cfg.output.cell           = [1] cell number of output data.
%
% cfg.method                = ['reversecorr']/'cca'/'cca2'. Reverse correlation, internal CCA
%                             implementation, or Telluride toolbox CCA implementation.
% cfg.zscore                = ['yes']/'no' whether to Z-score the training data. The Z-score
%                             normalization coefficients will be applied to the validation data
%                             and any test data evaluated with CO_DECODE_REGRESSION.
% cfg.weight                = [] 2-column weight vector with the same number of rows as the input
%                             data time dimension. The input data will be weighted with the first
%                             column, and the output data with the second column.
% cfg.gpu                   = ['no']/'yes' use the GPU if one is available.
%
%
% Reverse correlation options can be specified as follows:
%
% cfg.reversecorr.lags          = lags of the response function. Lags are always positive for causal
%                                 data.
% cfg.reversecorr.dir           = ['forward']/'backward' for forward (input causes output) /
%                                 backward (output causes input) mapping.
% cfg.reversecorr.regularize    = ['shrinkage']
%                                 'ridge'
%                                 'lra' low rank approximation.
%                                 'lasso' requires the GLMNET toolbox.
%                                 'admmlasso'
%                                 'none' no regularization, uses ordinary least squares.
% cfg.reversecorr.K             = (optional) regularization parameter(s)
%                                   shrinkage:  default=0.2, range [0:1]
%                                   ridge:      default=[10 0], ranges [0:Inf] [0 1]
%                                               K(2) specifies the order of the derivative of the
%                                               penalty (0: normal ridge, 1: first order derivative)
%                                               (see Lalor et al 2006: The VESPA) 
%                                   lra:        default =0.99, range [0:1]
%                                   lasso:      default =[.01 1], range [0:Inf, 0:1]
%                                               K(2) controls the degree of L1 penalty (elastic net)
% cfg.reversecorr.segments      = [1] number of segments in which to divide time-lagged data in
%                                 order to reduce memory usage.
%
%
% CCA options can be specified as follows:
%
% cfg.cca.thresh                = {1e-10,1e-10} threshold for whitening matrices for input and
%                                 output relative to the maximum eigenvalue. If greater than 1, this
%                                 will specify the number of eigenvalues kept. If empty (e.g.
%                                 {[],[]}), no pre-whitening will be performed. More finely
%                                 controlled whitening can be performed using CO_WHITENDATA.
% cfg.cca.reg                   = [0] CCA regularization value.
% cfg.cca.delay                 = [0] number of samples to delay the output with respect to the
%                                 input.
% cfg.cca.lags                  = {[],[]} 2-element cell array containing vectors of lags by which
%                                 the input and output data will be delayed.
% cfg.cca.segments              = [1] number of segments in which to divide time-lagged data in
%                                 order to reduce memory usage.
% cfg.cca.beamforming           = {'no','no'} whether to perform beamforming on input and/or output
%                                 during cross-validation.
% cfg.cca.beamsup               = [] indexes of CCA components to use for beamformer suppression.
% cfg.cca.beamsupext            = {[],[]} additional forward weights to suppress when beamforming is
%                                 used.
%
%
% Validation options can be specified to perform optimization of the first 
% cfg.reversecorr.K parameter or the cfg.cca.delay parameter. The  parameter is adjusted by
% cfg.validation.factor until termination conditions are met.
%
% cfg.validation.foldix     = {} sample indexes of cfg.dim for training/validation. Each cell
%                             specifies indexes for each validation fold. If empty, all indexes will
%                             be used for training/validation. If specified, but no validation takes
%                             place (cfg.validation.maxit=0), these sample indexes will still be
%                             used to compute the final decoder.
% cfg.validation.param      = Parameter to be optimized using cross-validation:
%                             ['K'] for cfg.method == 'reversecorr'.
%                             ['delay']/'thresh'/'lambda' for cfg.method == 'cca'.
% cfg.validation.factor     = [10] This is the factor by which cfg.validation.param will be adjusted.
%                             For reverse correlation:
%                               K(1) will be multiplied in each
%                               optimization step for ridge and lasso reverse correlation
%                               regularizations (default = 10). For shrinkage and lra, where the
%                               range is limited between 0 and 1, this number will be added to the
%                               logistically transformed K in order to maintain this range
%                               (default = 0.1).
%                             For CCA with cfg.validation.param set to:
%                               'thresh':   if cfg.cca.thresh is < 1, this factor will be added to
%                                           the logistically transformed cfg.cca.thresh to maintain
%                                           a rangebetween 0 and 1. (default = 0.1). Otherwise,
%                                           this factor will be added to cfg.cca.thresh
%                                           (default = 1).
%                               'reg':      cfg.cca.reg will be multiplied by this factor
%                                           (default = 10).
%                               'delay':    this factor will be added to cfg.cca.delay (default=1).
% cfg.validation.maxfail    = [0] This is the maximum number of iterations during which the
%                             validation result does not improve before termination of the
%                             optimization routine. The default of 0 signifies that the specified
%                             parameter will not be optimized. For CCA, it is recommended to use a
%                             larger number because the optimization curve is not smooth.
% cfg.validation.maxit      = [0] This is the maximum number of total iterations.
%
%
%
% data                      = training data. Both specified input and output must have the dimension
%                             specified in cfg.dim.
%
% OUTPUTS:
% decoder                   = decoder containing filter - filter has dimensions 
%                             [input_features (x output_features) x lags], where features is the 
%                             product of the length of non-time dimensions. The order of these 
%                             features is created by shifting the input so that the time dimension 
%                             is first, then reshaping the remaining dimensions into a single one.
%
% val_log
%
%
% See also: CO_DECODE_REGRESSION
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZH
% Author(s): Daniel D.E. Wong

if ~exist('FindTRF','file')
    disp('Adding the Telluride Decoding Toolbox to the MATLAB path.');
    addpath(fullfile(fileparts(which('co_defaults')), 'external', 'telluride-decoding-toolbox'));
end

if ~exist('nt_cov_lags','var')
    addpath(fullfile(fileparts(which('co_defaults')), 'external', 'NoiseTools'));
    addpath(fullfile(fileparts(which('co_defaults')), 'external', 'NoiseTools', 'COMPAT'));
end

assert(isfield(cfg,'input'),'No cfg.input specified.');
assert(isfield(cfg,'output'),'No cfg.output specified.');

% Set default options
if ~isfield(cfg,'dim'); cfg.dim = 'time'; end;
if ~isfield(cfg.input,'cell'); cfg.input.cell = 1; end;
if ~isfield(cfg.output,'cell'); cfg.output.cell = 1; end;
if ~isfield(cfg,'method'); cfg.method = 'reversecorr'; end;
if ~isfield(cfg,'zscore'); cfg.zscore = 'yes'; end;
if ~isfield(cfg,'weight'); cfg.weight = []; end;

infield = cfg.input.field;
outfield = cfg.output.field;
incell = cfg.input.cell;
outcell = cfg.output.cell;

cfgtmp = [];
cfgtmp.(infield) = [];
cfgtmp.(outfield) = [];
dim = co_checkdata(cfgtmp,data);

% Set cfg.dim to first dimension
indimix = find(strcmp(cfg.dim,dim.(infield))); %decoder.dim.indimsz(indimix) = 0;
outdimix = find(strcmp(cfg.dim,dim.(outfield))); %decoder.dim.outdimsz(outdimix) = 0;
cfgtmp = [];
cfgtmp.(infield).shift = indimix-1;
cfgtmp.(outfield).shift = outdimix-1;
data = co_shiftdim(cfgtmp,data);

% Initialize decoder structure
decoder = CoDecoder();
decoder.type = 'regression';
decoder.spec.regressalgo = cfg.method;
decoder.dim.dim = cfg.dim;
decoder.dim.indim = data.dim.(infield);
cfgtmp = []; cfgtmp.(infield).cell = incell; decoder.dim.indimsz = co_size(cfgtmp,data); %size(data.(infield){incell});
decoder.dim.indimsz(1) = 0;
decoder.dim.outdim = data.dim.(outfield);
cfgtmp = []; cfgtmp.(outfield).cell = outcell; decoder.dim.outdimsz = co_size(cfgtmp,data); %size(data.(outfield){outcell});
decoder.dim.outdimsz(1) = 0;

% Set more default options according to method type
if strcmp(cfg.method,'reversecorr')
    % Check reverse correlation options
    assert(isfield(cfg,'reversecorr'),'Reverse correlation options are missing.');
    assert(isfield(cfg.reversecorr,'lags'),'Missing cfg.reversecorr.lags');
    if ~isfield(cfg.reversecorr,'dir'); cfg.reversecorr.dir = 'forward'; end;
    if ~isfield(cfg.reversecorr,'regularize'); cfg.reversecorr.regularize = 'shrinkage'; end;
    if ~isfield(cfg.reversecorr,'K')
        switch cfg.reversecorr.regularize
            case 'shrinkage'
                cfg.reversecorr.K = 0.2;
            case 'ridge'
                cfg.reversecorr.K = [10,0];
            case 'lra'
                cfg.reversecorr.K = 0.99;
            case 'lasso'
                cfg.reversecorr.K = [0.01 1];
            case 'admmlasso'
                cfg.reversecorr.K = [0.01,1,1];
            otherwise
                cfg.reversecorr.K = 0;  % Validation optimization useless, but need to set a value
        end
    end
    if ~isfield(cfg.reversecorr,'segments'); cfg.reversecorr.segments = 1; end;
    decoder.spec.dir = cfg.reversecorr.dir;
    decoder.spec.lags = cfg.reversecorr.lags;
elseif strcmp(cfg.method,'cca') || strcmp(cfg.method,'cca2')
    assert(isfield(cfg,'cca') && isfield(cfg.cca,'delay'),'Missing cfg.cca.delay.');
    if ~isfield(cfg.cca,'thresh'); cfg.cca.thresh = {1e-10,1e-10}; end;
    if ~isfield(cfg.cca,'reg'); cfg.cca.reg = 0; end;
    if ~isfield(cfg.cca,'delay'); cfg.cca.delay = 0; end;
    if ~isfield(cfg.cca,'lags'); cfg.cca.lags = {[],[]}; end;
    if ~isfield(cfg.cca,'segments'); cfg.cca.segments = 1; end;
    if ~isfield(cfg.cca,'beamforming'); cfg.cca.beamforming = {'no','no'}; end;
    if ~isfield(cfg.cca,'beamsup'); cfg.cca.beamsup = []; end;
    if ~isfield(cfg.cca,'beamsupext'); cfg.cca.beamsupext = {[],[]}; end;
end

if ~isfield(cfg,'validation') || ~isfield(cfg.validation,'foldix'); cfg.validation.foldix = {}; end;
if ~isfield(cfg.validation,'param')
    if strcmp(cfg.method,'reversecorr')
        cfg.validation.param = 'K';
    else    % CCA
        cfg.validation.param = 'delay';
    end
end
if ~isfield(cfg.validation,'factor')
    if strcmp(cfg.method,'reversecorr')
        switch cfg.reversecorr.regularize
            case {'shrinkage','lra'}
                cfg.validation.factor = 0.01;
            otherwise
                cfg.validation.factor = 10;
        end
    else    % CCA
        cfg.validation.factor = 1;
    end
end
if ~isfield(cfg.validation,'maxfail'); cfg.validation.maxfail = 0; end;
if ~isfield(cfg.validation,'maxit'); cfg.validation.maxit = 0; end;

if ~isfield(cfg,'gpu'); cfg.gpu = 'no'; end;

assert(~strcmp(cfg.dim,'time') || data.fsample.(infield) == data.fsample.(outfield), ...
    'Input and output sample rates don''t match.');

indata = data.(infield){incell};
outdata = data.(outfield){outcell};

% Make indata and outdata 2 dimensional
szinshift = size(indata);
indata = reshape(indata,szinshift(1),prod(szinshift(2:end)));
szoutshift = size(outdata);
outdata = reshape(outdata,szoutshift(1),prod(szoutshift(2:end)));

% Determine indexes for validation
if isempty(cfg.validation.foldix)
    valix = {1:size(indata,1)};
else
    valix = cfg.validation.foldix;
end
valix_all = []; for ii = 1:length(valix); valix_all = [valix_all; valix{ii}(:)]; end;

if strcmp(cfg.gpu,'yes'); gpu = true; else gpu = false; end;

% Initialize parameter and score tracking for cross-validation optimization
if strcmp(cfg.method,'reversecorr')
    assert(strcmp(cfg.validation.param,'K'),'Unrecognized validation parameter');
    bestParam = cfg.reversecorr.K; lastParam = bestParam;
elseif strcmp(cfg.method,'cca') || strcmp(cfg.method,'cca2')
    switch cfg.validation.param
        case 'delay'
            bestParam = cfg.cca.delay;
        case 'reg'
            bestParam = cfg.cca.reg;
        case 'thresh'
            bestParam = cfg.cca.thresh;
        otherwise
            error('Unrecognized validation parameter');
    end
    lastParam = bestParam;
end
bestScore = 0;
valcounter = 0;
failcounter = 0; % Keeps track of optimization failures

val_log = {};
while 1
    % Check termination conditions
    if valcounter >= cfg.validation.maxit || failcounter >= cfg.validation.maxfail; break; end;
    
    fprintf('[*] Validation iteration %d...\n',valcounter+1);
    
    % Use cross validation to optimize specified parameter
    cr_val = [];    % Performance measure (correlation)
    cr_valm = 0; cr_valz = 0; cr_vals = 0;
    for cc = 1:length(valix)     % Cross-validation folds
        fprintf('\tCross validation fold %d of %d...',cc,length(valix));

        if length(valix) == 1
            trainix = valix{1};
        else
            trainix = setdiff(valix_all,valix{cc});
        end
        testix = valix{cc};
        
        % Z-score
        if strcmp(cfg.zscore,'yes')
            meanin = mean(indata(trainix,:),1); stdin = std(indata(trainix,:),[],1);
            meanout = mean(outdata(trainix,:),1); stdout = std(outdata(trainix,:),[],1);
        else
            meanin = zeros(1,size(indata,2)); stdin = ones(1,size(indata,2));
            meanout = zeros(1,size(outdata,2)); stdout = ones(1,size(outdata,2));
        end

        if strcmp(cfg.method,'reversecorr')
            % Determine current parameter
            if valcounter > 0
                switch cfg.reversecorr.regularize
                    case {'ridge','lasso'}
                        currParam(1) = lastParam(1) * cfg.validation.factor;
                    case {'shrinkage','lra','admmlasso'}  % Keep in range [0 1]
                        currParam(1) = logsig(log(lastParam(1)) - log(1-lastParam(1)) + ...
                            cfg.validation.factor);
                end
            else
                currParam = lastParam;
            end
            
            % Compute reverse correlation
            if strcmp(cfg.reversecorr.dir,'forward')
                stimulus = (indata(trainix,:)-repmat(meanin,length(trainix),1))./repmat(stdin,length(trainix),1);
                response = (outdata(trainix,:)-repmat(meanout,length(trainix),1))./repmat(stdout,length(trainix),1);
                dir = 1;
            else
                stimulus = (outdata(trainix,:)-repmat(meanout,length(trainix),1))./repmat(stdout,length(trainix),1);
                response = (indata(trainix,:)-repmat(meanin,length(trainix),1))./repmat(stdin,length(trainix),1);
                dir = -1;
            end
            if ~isempty(cfg.weight)
                stimulus = stimulus.*repmat(cfg.weight(trainix,1),1,size(stimulus,2));
                response = response.*repmat(cfg.weight(trainix,2),1,size(stimulus,2));
            end
            test_in = (indata(testix,:)-repmat(meanin,length(testix),1))./repmat(stdin,length(testix),1);
            test_out = (outdata(testix,:)-repmat(meanout,length(testix),1))./repmat(stdout,length(testix),1);
            [~,pred] = FindTRF(stimulus,response,dir,test_in,[],cfg.reversecorr.lags, ...
                cfg.reversecorr.regularize,currParam,0,[],cfg.reversecorr.segments,gpu);

            % Determine performance
            cr = zeros(1,size(pred,2));
            for ii = 1:size(pred,2)
                cr(ii) = corr2(test_out(:,ii),pred(:,ii));
            end
        elseif strcmp(cfg.method,'cca') || strcmp(cfg.method,'cca2')
            EXP=1-10^-12;   % Used to break symmetry when x and y are perfectly correlated
            
            if valcounter > 0
                switch cfg.validation.param
                    case 'thresh'   % Keep in range [0 1]
                        currParam(1) = logsig(log(lastParam(1)) - log(1-lastParam(1)) + ...
                            cfg.validation.factor);
                    case 'reg'
                        currParam(1) = lastParam(1) * cfg.validation.factor;
                    case 'delay'
                        currParam = lastParam + cfg.validation.factor;
                end
            else
                currParam = lastParam;
            end
            
            % Determine CCA parameters for current iteration
            paramnames = {'delay','reg','thresh'};
            param = {cfg.cca.delay,cfg.cca.reg,cfg.cca.thresh};
            param{strcmp(paramnames,cfg.validation.param)} = currParam;
            
            x = (indata(trainix,:)-repmat(meanin,length(trainix),1))./repmat(stdin,length(trainix),1);
            y = (outdata(trainix,:)-repmat(meanout,length(trainix),1))./repmat(stdout,length(trainix),1);
            if ~isempty(cfg.weight); x = x.*repmat(cfg.weight(trainix,1),1,size(x,2)); y = y.*repmat(cfg.weight(trainix,2),1,size(y,2)); end;
            if strcmp(cfg.method,'cca') % NoiseTools version
                [A,B] = cca_nt(x,y,param{:},cfg.cca.segments,gpu);
            else                        % Telluride toolbox version
                [A,B] = cca_td(x,y,param{:},cfg.cca.segments,gpu);
            end
            N = min(size(A,2),size(B,2));
            
            % Beamforming
            if strcmp(cfg.cca.beamforming{1},'yes') % Input
                A = A./repmat(sqrt(sum(A.^2,1)),size(A,1),1);
                A = beamforming(indata(testix,:),pinv(nt_normcol(A))',N,cfg.cca.beamsup,cfg.cca.beamsupext{1});
%                 A = beamforming(indata(testix,:)*P{1},invVt(1:size(A,1),1:size(A,2)),N,cfg.cca.beamsup,cfg.cca.beamsupext{1});
%                 A = P{1}*A;
            end
            if strcmp(cfg.cca.beamforming{2},'yes')	% Output
                B = beamforming(outdata(testix,:),pinv(nt_normcol(B))',N,cfg.cca.beamsup,cfg.cca.beamsupext{2});
%                 B = beamforming(outdata(testix,:)*P{2},invVt(size(A,1)+1:end,size(A,2)+1:end),N,cfg.cca.beamsup,cfg.cca.beamsupext{2});
%                 B = P{1}*B;
            end

            % Compute sum of component correlations (want to maximize this)
            cc1 = indata(testix,:)*A(:,1:N);
            cc2 = outdata(testix,:)*B(:,1:N);
            if strcmp(cfg.validation.param,'delay')
                [cc1,cc2]=nt_relshift(cc1,cc2,currParam);
            else
                [cc1,cc2]=nt_relshift(cc1,cc2,cfg.cca.delay);
            end
            cr = zeros(1,size(cc1,2));
            for ii = 1:size(cc1,2)
                C = corrcoef(cc1(:,ii),cc2(:,ii));
                cr(ii) = C(1,2);
            end
        end
        fprintf('performance: %d\n',sum(cr));
        if isempty(cr_val)
            cr_val = cr;
        else
            cr_val = [cr_val; cr];
        end
        %cr_val = cr_val+cr;
    end
    cr_valm = mean(cr_val,1); %cr_val./length(valix); % Average over # of validation folds
    cr_vals = sum(cr_valm);
    cr_valz = cr_valm./std(cr_val,[],1);
    
    % Update bestScore/Param tracking and counters
    if cr_vals > bestScore
        bestScore = sum(cr_valm);
        bestParam = currParam;
    else
        failcounter = failcounter + 1;
    end
    valcounter = valcounter + 1;
    
    fprintf('\tPerformance for iteration %d: %d (best score: %d)\n',...
        valcounter,sum(cr_valm),bestScore);

    lastParam = currParam;  % For next optimization iteration
    val_log{end+1}.param = currParam;
    val_log{end}.score = cr_vals;
    val_log{end}.cr_valm = cr_valm;
    val_log{end}.cr_valz = cr_valz;
    val_log{end}.bestparam = bestParam;
    val_log{end}.bestscore = bestScore;
end

% Use best parameter to compute decoder with all specified data indexes
% Z-score
if strcmp(cfg.zscore,'yes')
    meanin = mean(indata(valix_all,:),1); stdin = std(indata(valix_all,:),[],1);
    meanout = mean(outdata(valix_all,:),1); stdout = std(outdata(valix_all,:),[],1);
else
    meanin = zeros(1,size(indata,2)); stdin = ones(1,size(indata,2));
    meanout = zeros(1,size(outdata,2)); stdout = ones(1,size(outdata,2));
end
decoder.spec.meanin = meanin;
decoder.spec.stdin = stdin;
decoder.spec.meanout = meanout;
decoder.spec.stdout = stdout;

% Compute decoder for output
if strcmp(cfg.method,'reversecorr')
    if strcmp(cfg.reversecorr.dir,'forward')
        stimulus = (indata(valix_all,:)-repmat(meanin,length(valix_all),1))./repmat(stdin,length(valix_all),1);
        response = (outdata(valix_all,:)-repmat(meanout,length(valix_all),1))./repmat(stdout,length(valix_all),1);
        dir = 1;
    else
        stimulus = (outdata(valix_all,:)-repmat(meanout,length(valix_all),1))./repmat(stdout,length(valix_all),1);
        response = (indata(valix_all,:)-repmat(meanin,length(valix_all),1))./repmat(stdin,length(valix_all),1);
        dir = -1;
    end
    if ~isempty(cfg.weight)
        stimulus = stimulus.*repmat(cfg.weight(valix_all,1),1,size(stimulus,2));
        response = response.*repmat(cfg.weight(valix_all,2),1,size(response,2));
    end

    g = FindTRF(stimulus,response,dir,[],[],cfg.reversecorr.lags, ...
        cfg.reversecorr.regularize,bestParam,0,[],cfg.reversecorr.segments,gpu);

    decoder.decoder = g;
    decoder.spec.regularize = cfg.reversecorr.regularize;
    if strcmp(cfg.dim,'time'); decoder.spec.fsample = data.fsample.(cfg.input.field); end;
else
    % Determine CCA parameters
    paramnames = {'delay','reg','thresh'};
    param = {cfg.cca.delay,cfg.cca.reg,cfg.cca.thresh};
    param{strcmp(paramnames,cfg.validation.param)} = bestParam;
    
    x = (indata(valix_all,:)-repmat(meanin,length(valix_all),1))./repmat(stdin,length(valix_all),1);
    y = (outdata(valix_all,:)-repmat(meanout,length(valix_all),1))./repmat(stdout,length(valix_all),1);
    if ~isempty(cfg.weight); x = x.*repmat(cfg.weight(valix_all,1),1,size(x,2)); y = y.*repmat(cfg.weight(valix_all,2),1,size(y,2)); end;
    
    if strcmp(cfg.method,'cca')
        [A,B,R,invA,invB] = cca_nt(x,y,param{:},cfg.cca.segments,gpu);
        decoder.decoder.invA = invA; decoder.decoder.invB = invB;
    else
        [A,B,R] = cca_td(x,y,param{:},cfg.cca.segments,gpu);
    end
    
    decoder.decoder.A = A;
    decoder.decoder.B = B;
    decoder.decoder.R = R;
%     decoder.decoder.invVt = invVt;
%     decoder.decoder.P = P;
    for ii = 1:length(paramnames); decoder.spec.(paramnames{ii}) = param{ii}; end;
end


function [A,B,R,invA,invB] = cca_nt(x,y,delay,reg,thresh,segs,gpu)
% Performs CCA - Adapted from nt_cca in NoiseToolbox

EXP=1-10^-12;   % Used to break symmetry when x and y are perfectly correlated

% Covariance matrices
R = cell(1,2); R_all = [];
seglen = ceil(size(x,1)/segs);
for ii = 1:segs
    ix = (ii-1)*seglen+1; ix = ix:min(ix+seglen-1,size(x,1));
    [R_all_tmp,~,m] = nt_cov_lags(x(ix,:),y(ix,:),delay); %R_all = R_all + reg*eye(size(R_all));
    if isempty(R{1})
        R{1} = R_all_tmp(1:m,1:m);
        R{2} = R_all_tmp(m+1:end,m+1:end);
        R_all = R_all_tmp;
    else
        R{1} = R{1} + R_all_tmp(1:m,1:m);
        R{2} = R{2} + R_all_tmp(m+1:end,m+1:end);
        R_all = R_all + R_all_tmp;
    end
end
R{1} = R{1} + reg*eye(size(R{1})); R{2} = R{2} + reg*eye(size(R{2}));
R_all = R_all + reg*eye(size(R_all));

% Compute whitening matrices
P = cell(1,2);
for ii = 1:2
    if ~isempty(thresh{ii})
        R{ii} = gpuConvert(R{ii},gpu);
        [V,D] = eig(R{ii}); V = real(V); D = real(D); D = diag(D);
        [D,ix] = sort(D,'descend'); V = V(:,ix);
        V = gpuGather(V,gpu); D = gpuGather(D,gpu);

        if thresh{ii} > 1
            assert(thresh{ii}<=length(D), ...
                ['cfg.cca.thresh{' num2str(ii) '} exceeds the number of components.']);
            ix = 1:thresh{ii};
        else
            ix = find(D/max(D) > thresh{ii});
        end
        P{ii} = V(:,ix)*diag(sqrt(1./D(ix).^EXP));
    else
        P{ii} = eye(size(R{ii},1));
    end
end

% Apply whitening
blkP = blkdiag(P{1},P{2});
Rwt = blkP'*R_all*blkP;
%N = min(size(P{1},2),size(P{2},2)); % Number of CCA components

% Do CCA
Rwt = gpuConvert(Rwt,gpu);
[V,D] = eig(Rwt); V = real(V); D = real(D); D = diag(D);
[R,ix] = sort(D,'descend'); V = V(:,ix);
V = gpuGather(V,gpu); R = gpuGather(R,gpu);
% invVt = pinv(V)';
V = blkP*V;
A = V(1:size(P{1},1),1:size(P{1},2))*sqrt(2); B = V(size(P{1},1)+1:end,1:size(P{2},2))*sqrt(2);

invV = pinv(V)';
invA = invV(1:size(P{1},1),1:size(P{1},2))*sqrt(2); invB = invV(size(P{1},1)+1:end,1:size(P{2},2))*sqrt(2);


function [A,B,R] = cca_td(x,y,delay,reg,thresh,segs,gpu)
EXP=1-10^-12;   % Used to break symmetry when x and y are perfectly correlated
[x,y] = nt_relshift(x,y,delay);

% Compute whitening matrices
P = cell(1,2);
R = cell(1,2); R{1} = x'*x; R{2} = y'*y;
for ii = 1:2
    if ~isempty(thresh{ii})
        R{ii} = gpuConvert(R{ii},gpu);
        [V,D] = eig(R{ii}); V = real(V); D = real(D); D = diag(D);
        [D,ix] = sort(D,'descend'); V = V(:,ix);
        V = gpuGather(V,gpu); D = gpuGather(D,gpu);

        if thresh{ii} > 1
            assert(thresh{ii}<=length(D), ...
                ['cfg.cca.thresh{' num2str(ii) '} exceeds the number of components.']);
            ix = 1:thresh{ii};
        else
            ix = find(D/max(D) > thresh{ii});
        end
        P{ii} = V(:,ix)*diag(sqrt(1./D(ix).^EXP));
    else
        P{ii} = eye(size(R{ii},1));
    end
end
x = x*P{1}; y = y*P{2};
[A,B,R] = cca(x',y',reg,segs);
A = P{1}*A; B = P{2}*B;


function W = beamforming(x,L,N,beamsup,beamsupext)
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

R = x'*x;
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



function X = gpuConvert(X,gpu)
% Helper function for getting variables onto the GPU
if gpu && exist('gpuDeviceCount','file') && gpuDeviceCount() > 0
    X=gpuArray(X);
end

function X = gpuGather(X,gpu)
% Helper function for getting variables off GPU
if gpu && exist('gather','file') && gpuDeviceCount() > 0
    X=gather(X);
end
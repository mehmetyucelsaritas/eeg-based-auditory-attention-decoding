function components = co_compute_components(cfg,data)
% CO_COMPUTE_COMPONENTS performs a component analysis on the data and returns the mixing and
% unmixing matrices, as well as the powers of each component. The rows in the mixing/unmixing
% matrices correspond to the mixed dimensions (e.g. channels), while the columns correspond to the
% components. For CSP/CSPOVA/CSPN methods, different classes are specified using both cfg.FIELD.cell
% and cfg.FIELD.samples.
%
% INPUTS:
% cfg.FIELD.method      = 'pca' (principal component analysis) separates signal into linearly
%                               uncorrelated components.
%                         'ica' (independent component analysis) separates signal into statistically
%                               independent components.
%                         'dss' (denoising source separation)
%                         'cca' (cannonical component analysis) can handle N-classes.
%                         'csp' (common spatial patterns) separates signal into components which
%                               have maximum variance between two specified classes.
%                         'cspova' performs a CSP on multiple classes using a one-vs-all scheme. The
%                               weighting of classes in the 'all' group is defined in
%                               cfg.FIELD.cspweight.
%                         'cspn' multiclass version of CSP.
%
% cfg.FIELD.dss         = options for DSS
%               .weight = vector containing weights for weighting the operating dimension (specified
%                         by cfg.FIELD.dim) when computing the DSS covariance matrix. Must have same
%                         length as the selected samples in the operating dimension (see
%                         cfg.FIELD.samples).
%               .thresh = [1e-10]
%
% cfg.FIELD.cca         = options for CCA
%               .reg    = [0] ridge regularization value.
%               .thresh = {1e-10,...}
%
% cfg.FIELD.csp         = options for CSP, CSPOVA, or CSPN
%               .reg    = [0] CSP ridge regularization value (for CSP or CSPOVA). It is worth noting
%                         that regularization can also be accomplished by adjusting the
%                         pre-whitening matrix eigenvalue threshold (see cfg.FIELD.csp.thresh).
%               .thresh = [1e-10] threshold for pre-whitening matrix eigenvalues relative
%                         to the largest eigenvalue. If greater than 1, this will specify the number
%                         of eigenvalues kept. If empty, no pre-whitening will be performed. More
%                         finely controlled whitening can be performed using CO_WHITENDATA.
%               .weight = the weighting matrix for the contribution of classes to the 'all' group
%                         for the 'cspova' method. Each row corresponds to the 'one' group and the
%                         columns correspond to the weightings of the classes in the 'all' group.
%                         Diagonals elements, as such are not used. This defaults to a square matrix
%                         of ones, with side-lengths equal to the number of classes.
%
% cfg.FIELD.dim         = ['time'] dimension on which to operate.
% cfg.FIELD.cell        = [1] index of data trial/cell to use to compute components. If CSP or CCA
%                         are used, this should be an array with one element per class. The samples
%                         used for each class should be specified in cfg.FIELD.samples.
% cfg.FIELD.samples     = [{'all'}]/cell array with one cell per class. Each cell element contains 
%                         an array of sample indexes to use to compute components.
% cfg.FIELD.segments    = number of segments in which to divide data when computing covariance
%                         matrix.
%
% data
%
%
% OUTPUTS:
% components        = structure with the following fields:
%   .FIELD.A        = mixing matrix. Sorted from largest to smallest component. Components are along
%                     the last dimension.
%   .FIELD.W        = unmixing matrix. Sorted from largest to smallest component. Components are
%                     along the last dimension.
%   .FIELD.D        = component powers, or in the case of CSPN, component mutual information.
%
%
% See also: CO_DIMREDUCTION, CO_WHITENDATA
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZH
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);

components = struct();
for ii = 1:length(fields)
    if ~isfield(cfg.(fields{ii}),'dim'); cfg.(fields{ii}).dim = 'time'; end;
    if ~isfield(cfg.(fields{ii}),'cell'); cfg.(fields{ii}).cell = 1; end;
    if ~isfield(cfg.(fields{ii}),'samples')
        cfg.(fields{ii}).samples = cell(1,length(cfg.(fields{ii}).cell));
        for jj = 1:length(cfg.(fields{ii}).samples); cfg.(fields{ii}).samples{jj} = {'all'}; end;
    end
    assert(iscell(cfg.(fields{ii}).samples'), ['cfg.' fields{ii} '.samples must be a cell array.']);
    
    assert(isfield(cfg.(fields{ii}),'method'), ['cfg.' fields{ii} '.method not specified']);
    if strcmp(cfg.(fields{ii}).method,'cca'); cfg.(fields{ii}).cca = struct(); end;
    if strcmp(cfg.(fields{ii}).method,{'csp','cspova','cspn'}); cfg.(fields{ii}).csp = struct(); end;
    if isfield(cfg.(fields{ii}),'cca')
        if ~isfield(cfg.(fields{ii}).cca,'reg'); cfg.(fields{ii}).cca.reg = 0; end;
        if ~isfield(cfg.(fields{ii}).cca,'thresh')
            cfg.(fields{ii}).cca.thresh = cell(1,length(cfg.(fields{ii}).cell));
            for jj = 1:length(cfg.(fields{ii}).cell); cfg.(fields{ii}).cca.thresh{jj} = 1e-10; end;
        end
    end
    
    if ~isfield(cfg.(fields{ii}),'csp'); cfg.(fields{ii}).csp = []; end;  
    if ~isfield(cfg.(fields{ii}).csp,'reg'); cfg.(fields{ii}).csp.reg = 0; end;
    if ~isfield(cfg.(fields{ii}).csp,'thresh'); cfg.(fields{ii}).csp.thresh = 1e-10; end;
    if ~isfield(cfg.(fields{ii}).csp,'weight'); cfg.(fields{ii}).csp.weight = ...
        ones(length(cfg.(fields{ii}).cell)); end;

    if ~isfield(cfg.(fields{ii}),'segments'); cfg.(fields{ii}).segments = 1; end;
    
    % Separate the different classes into cells
    datatmp = cell(1,length(cfg.(fields{ii}).cell));
    dimix = find(strcmp(cfg.(fields{ii}).dim, dim.(fields{ii})));
    sz = cell(1,length(cfg.(fields{ii}).cell));
    for jj = 1:length(cfg.(fields{ii}).cell)
        % Make cfg.FIELD.dim the first dimension
        datatmp{jj} = shiftdim(data.(fields{ii}){cfg.(fields{ii}).cell(jj)},dimix-1);
    
        % Make datatmp 2D
        sz{jj} = size(datatmp{jj});
        datatmp{jj} = reshape(datatmp{jj},sz{jj}(1),prod(sz{jj}(2:end)));
    
        if isnumeric(cfg.(fields{ii}).samples{jj}); datatmp{jj} = datatmp{jj}(cfg.(fields{ii}).samples{jj},:); end;
        
        if ~strcmp(cfg.(fields{ii}).method,'cca') && jj > 1
            assert(size(datatmp{jj},2)==size(datatmp{1},2),['Total size of non-' ...
                    cfg.(fields{ii}).dim ' dimensions is not the same between specified cells.']);
        end
    end
    
    switch cfg.(fields{ii}).method
        case 'pca'
            assert(length(datatmp)==1,'Too many classes defined for PCA.');
            [W, D] = eig(covseg(datatmp{1},cfg.(fields{ii}).segments)); D = diag(D);
            [D,ix] = sort(D,'descend'); W = W(:,ix);
            A = pinv(W)';
        case 'ica'
            assert(length(datatmp)==1,'Too many classes defined for ICA.');
            [S,A] = icaML(datatmp{1}');
            Ainv = pinv(A)';
            D = sqrt(sum(S.^2,2))'./sqrt(sum(Ainv.^2,1)); [D,ix] = sort(D,'descend');
            W = Ainv./repmat(sqrt(sum(Ainv.^2,1)),size(Ainv,1),1); W = W(:,ix);
            A = A(:,ix);
        case 'dss'
            assert(length(datatmp)==1,'Too many classes defined for DSS.');
            R = covseg(datatmp{1},cfg.(fields{ii}).segments);
            R1 = datatmp{1}.*repmat(cfg.(fields{ii}).dss.weight(:),1,size(datatmp{1},2));
            R1 = R1'*R1;
            R = R./trace(R); R1 = R1./trace(R1);    % Normalize
            
            % Prewhiten
            if ~isempty(cfg.(fields{ii}).dss.thresh)
                [V,D] = eig(R);
                V = real(V); D = real(D);
                [D,ix] = sort(diag(D),'descend'); V = V(:,ix);
                if cfg.(fields{ii}).dss.thresh > 1
                    assert(cfg.(fields{ii}).dss.thresh<=length(D), ...
                        'cspthresh exceeds the number of components.');
                    ix = 1:cfg.(fields{ii}).dss.thresh;
                else
                    ix = find(D/max(D) > cfg.(fields{ii}).dss.thresh);
                end
                D = D(ix); V = V(:,ix);
                P = diag(sqrt(1./D))*V';
                
                R = P*R*P';
                R1 = P*R1*P';
            end
            
            % DSS
            [V,D] = eig(R1,R);
            V = real(V); D = real(diag(D));
            [D,ix]=sort(abs(D),'descend'); V=V(:,ix);
            W = V; if exist('P','var'); W = P'*W; end;
            A = pinv(W)';
        case 'cca'
            assert(length(datatmp)>1,'More than 1 class must be defined for CCA.');
            [A,W,D] = cca(datatmp,cfg.(fields{ii}).cca.reg,cfg.(fields{ii}).cca.thresh,cfg.(fields{ii}).segments);
        case 'csp'
            assert(length(datatmp)==2,'2 classes must be defined for CSP.');
            R1 = covseg(datatmp{1},cfg.(fields{ii}).segments); R1 = R1/trace(R1);
            R2 = covseg(datatmp{2},cfg.(fields{ii}).segments); R2 = R2/trace(R2);
            [A,W,D] = csp(R1,R2,cfg.(fields{ii}).csp.reg,cfg.(fields{ii}).csp.thresh);
        case 'cspova'
            assert(length(datatmp)>2,'More than 2 classes must be defined for CSP one-vs-all.');
            W = cell(1,length(datatmp)); A = cell(1,length(datatmp)); D = cell(1,length(datatmp));
            for jj = 1:length(datatmp)
                R1 = covseg(datatmp{jj},cfg.(fields{ii}).segments); R1 = R1/trace(R1);
                R2 = zeros(size(R1));
                for kk = 1:length(datatmp)
                    if kk == jj; continue; end;
                    R2 = R2 + covseg(datatmp{kk},cfg.(fields{ii}).segments).*cfg.(fields{ii}).csp.weight(jj,kk);
                end
                R2 = R2/trace(R2);
                [A{jj},W{jj},D{jj}] = csp(R1,R2,cfg.(fields{ii}).csp.reg,cfg.(fields{ii}).csp.thresh);
            end
        case 'cspn'
            assert(length(datatmp)>=2,'2 or more classes must be defined for CSPN.');
            R = zeros(length(datatmp),size(datatmp{1},2),size(datatmp{1},2));
            for jj = 1:length(datatmp)
                R(jj,:,:) = covseg(datatmp{jj},cfg.(fields{ii}).segments);
                R(jj,:,:) = R(jj,:,:)/trace(squeeze(R(jj,:,:)));
            end
            % Prewhiten
            if ~isempty(cfg.(fields{ii}).csp.thresh)
                [V,D] = eig(squeeze(sum(R,1))); V = real(V); D = real(D); D = diag(D);
                [D,ix] = sort(D,'descend'); V = V(:,ix);
                if cfg.(fields{ii}).csp.thresh > 1
                    assert(cfg.(fields{ii}).csp.thresh<=length(D), ...
                        'csp.thresh exceeds the number of components.');
                    ix = 1:cfg.(fields{ii}).csp.thresh;
                else
                    ix = find(D/max(D) > cfg.(fields{ii}).csp.thresh);
                end
                D = D(ix); V = V(:,ix);
                P = diag(sqrt(1./D))*V';
                S = zeros([size(R,1), length(D), length(D)]);
                for jj = 1:size(R,1)
                    S(jj,:,:) = P*squeeze(R(jj,:,:))*P';
                end
            else
                S = R;
            end
            
            [W,D] = MulticlassCSP(S,size(S,3));
            if exist('P','var'); W = P'*W'; else W = W'; end;
            A = pinv(W)';
        otherwise
            error(['Unknown cfg.' fields{ii} '.method']);
    end
    
    % Reshape W and A
    if iscell(W)    % Check if cell to handle cspova and cca
        for jj = 1:length(W)  
            W{jj} = reshape(W{jj},[sz{jj}(2:end), size(W{jj},2)]);
            A{jj} = reshape(A{jj},[sz{jj}(2:end), size(A{jj},2)]);
        end
    else
        W = reshape(W,[sz{jj}(2:end), size(W,2)]); A = reshape(A,[sz{jj}(2:end), size(A,2)]);
    end
    
    components.(fields{ii}).A = A;
    components.(fields{ii}).W = W;
    components.(fields{ii}).D = D;
end


function [A,W,R] = cca(y, reg, thresh, seg)
% Performs CCA - Adapted from nt_cca in NoiseToolbox

EXP=1-10^-12;   % Used to break symmetry when x and y are perfectly correlated

% Covariance matrices
sz_ycat = zeros(1,2); sz_ycat(1) = size(y{1},1); m = zeros(1,length(y));
for ii = 1:length(y); m(ii) = size(y{ii},2); sz_ycat(2) = sz_ycat(2) + m(ii); end;
y_cat = zeros(sz_ycat);
for ii = 1:length(y)
    endix = sum(m(1:ii)); startix = endix-m(ii)+1;
    y_cat(:,startix:endix) = y{ii};
end

R_all = covseg(y_cat,seg); R_all = R_all + reg*eye(size(R_all));
R = cell(1,length(y));
for ii = 1:length(y)
    endix = sum(m(1:ii)); startix = endix-m(ii)+1;
    R{ii} = R_all(startix:endix,startix:endix);
end

% Compute whitening matrices
P = cell(1,length(y));
for ii = 1:length(y)
    if ~isempty(thresh{ii})
        [V,D] = eig(R{ii}); V = real(V); D = real(D); D = diag(D);
        [D,ix] = sort(D,'descend'); V = V(:,ix);

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
blkP = blkdiag(P{:});
Rwt = blkP'*R_all*blkP;
%N = min(size(P{1},2),size(P{2},2)); % Number of CCA components

% Do CCA
[V,D] = eig(Rwt); V = real(V); D = real(D); D = diag(D);
[D,ix] = sort(D,'descend'); V = V(:,ix);
A = cell(1,length(y)); W = cell(1:length(y)); R = cell(1:length(y));
endix = 0;
for ii = 1:length(y)
    startix = endix+1; endix = endix+size(P{ii},1);
    W{ii} = P{ii}*V(startix:endix,1:size(P{ii},1))*sqrt(2);
    A{ii} = pinv(W{ii})';   % Is this right?
    R{ii} = D(startix:endix);
end


function [A,W,D] = csp(R1,R2,cspreg,cspthresh)
assert(isempty(cspthresh) || cspreg == 0, 'Combined whitening and regularization not currently supported.');

% CSP function
if ~isempty(cspthresh)
    R = R1 + R2;                            % Combined covariance matrix
    [V,D] = eig(R); V = real(V); D = real(D); D = diag(D);
    [D,ix] = sort(D,'descend'); V = V(:,ix);
    if cspthresh > 1
        assert(cspthresh<=length(D),'cspthresh exceeds the number of components.');
        ix = 1:cspthresh;
    else
        ix = find(D/max(D) > cspthresh);
    end
    D = D(ix); V = V(:,ix);
    P = diag(sqrt(1./D))*V';
    S1 = P*R1*P'; S2 = P*R2*P';             % Apply whitening matrix
else
    S1 = R1; S2 = R2;
end

if cspreg == 0         % No regularization
    [V,D] = eig(S1,S2); D = diag(D);
%     [V1,D1] = eig(S1,S2+S1);    % CCACSP variation - Proposed by Noh, 2013
%     [V2,D2] = eig(S2,S2+S1);
%     V = [V1 V2]; D = [diag(D1); diag(D2)];
else                                    % Tikhonov regularization
    [V1,D1] = eig(inv(S2+eye(size(S2))*cspreg)*S1);
    [V2,D2] = eig(inv(S1+eye(size(S1))*cspreg)*S2);
    V = [V1 V2]; D = [diag(D1); diag(D2)];
end
V = real(V); D = real(D);
[D,ix]=sort(D,'descend'); V=V(:,ix);
if cspreg == 0; [D,ix]=sort(max([D(:), 1./D(:)],[],2),'descend'); V=V(:,ix); end;   % D is a ratio
if exist('P','var');
    W = P'*V;
    W = W./repmat(sqrt(sum(W.^2,1)),size(W,1),1);    % Normalize W so A can be computed
else
    W = V;
end
A = pinv(W)';



function R = covseg(x,segs)
R = zeros(size(x,2));
seglen = ceil(size(x,1)/segs);
for ii = 1:segs
    ix = (ii-1)*seglen+1; ix = ix:min(ix+seglen-1,size(x,1));
    R = R+x(ix,:)'*x(ix,:);
end
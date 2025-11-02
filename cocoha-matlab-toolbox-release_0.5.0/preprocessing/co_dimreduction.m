function data = co_dimreduction(cfg,data)
% CO_DIMREDUCTION reduces the dimensions of the specified data field using the specified
% unmixing/mixing matrices. If the data has more than 2 dimensions, it will be reshaped to 2
% dimensions before applying dimension reduction. Labels for the reduced dimensions will be removed.
% If the time dimension happens to be one of the reduced dimensions, the events field will be
% removed. The reduced dimension will be renamed to 'comp'. This operation is performed for all
% cells.
%
% INPUTS:
% cfg.FIELD.dim         = ['time'] dimension which will be left untouched by the dimension
%                         reduction.
% cfg.FIELD.method      = ['unmixing']/'beamforming'. Unmixing simply multiplies the data with the
%                         specified columns of the unmixing matrix to obtain the components.
%                         Beamforming applies an LCMV beamformer to the data, using the specified
%                         mixing matrix components as the forward solution. This implementation
%                         assumes that mixing matrix components are orthogonal to each other.
%                         Beamforming may be useful for datasets that may have a different
%                         distribution of noise sources than that used to compute the components.
% cfg.FIELD.components  = {}, a cell array containing arrays of component indexes, one cell for each
%                         data cell/trial. The component indexes should refer to the column index of
%                         the unmixing/mixing matrix. If unspecified, all columns will be used.
% cfg.FIELD.beamsup     = beamforming suppression group specification array. LCMV will project unit
%                         gain on components specified in cfg.FIELD.components while enforcing nulls
%                         on all other components in the same group.
%                         This is a cell array containing a numerical array per data cell/trial. The
%                         numerical array should have the same length as the number of component
%                         indexes in cfg.FIELD.components{cell}. Array elements with the same number
%                         indicate that the corresponding components will be suppressed concurrently
%                         during beamforming. Default is to suppress all components at the same
%                         time.
% cfg.FIELD.beamix      = cell array with one cell per data cell, indicating indexes along
%                         specified dimension over which beamformer covariance matrix is to be
%                         computed for each cell. If no indexes are specified, all indexes will be
%                         used. Default is to use all indexes.
% cfg.FIELD.beamopt
%               .type   = ['lcmv']/'lcmveig' type of beamformer. 'lcmv' performs LCMV, before
%                         performing Borgiotti-Kaplan style weight normalization. 'eiglcmv'
%                         uses an eigenvalue LCMV beamformer before performing Borgiotti-Kaplan
%                         style weight normalization (Sekihara, 2002).
%               .lcmveig  options for the lcmveig beamformer.
%                   .ns = [1] number of eigenvalues in signal space for the eigenvalue beamformer
%                         ('lcmveig').
%
% If method is 'unmixing, the following must be specified:
% cfg.FIELD.W           = A cell array containing the unmixing matrices in each cell, one for each
%                         data cell/trial.
%
% If method is 'beamforming', the following must be specified:
% cfg.FIELD.A           = A cell array containing the mixing matrices in each cell, one for each
%                         data cell/trial.
%
% cfg.FIELD.F           = (optional) a cell array containing an FIR filter for each
%                         unmixing/mixing matrix component in each cell, one for each data
%                         cell/trial. Filter will work along dimension specified by cfg.FIELD.dim.
%
% data
%
% OUTPUTS:
% data
%
%
% See also: CO_COMPUTE_COMPONENTS
% K. Sekihara, S. S. Nagarajan, D. Poeppel, A. Marantz, and Y. Miyashita, "Application of an MEG
% eigenspace beamformer to reconstructing spatio-temporal activities of neural sources," Hum. Brain
% Map.
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU,Oticon, UCL, UZ
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);

for ii = 1:length(fields)
    % Defaults
    if ~isfield(cfg.(fields{ii}),'dim'); cfg.(fields{ii}).dim = 'time'; end;
    if ~isfield(cfg.(fields{ii}),'method'); cfg.(fields{ii}).method = 'unmixing'; end;
    if ~isfield(cfg.(fields{ii}),'components'); cfg.(fields{ii}).components = {}; end;
    if ~isfield(cfg.(fields{ii}),'beamix')
        cfg.(fields{ii}).beamix = cell(1,length(data.(fields{ii})));
    end
    if ~isfield(cfg.(fields{ii}),'beamopt'); cfg.(fields{ii}).beamopt.type = 'lcmv'; end;
    if ~isfield(cfg.(fields{ii}).beamopt,'lcmveig') || ~isfield(cfg.(fields{ii}).beamopt.lcmveig,'n')
        cfg.(fields{ii}).beamopt.lcmveig.n = 1;
    end
    
    % Asserts
    switch cfg.(fields{ii}).method
        case 'unmixing'
            assert(isfield(cfg.(fields{ii}),'W'), ['Missing cfg.' fields{ii} '.W for unmixing method.']);
            assert(iscell(cfg.(fields{ii}).W), ['cfg.' fields{ii} '.W must be a cell array.']);
            assert(length(cfg.(fields{ii}).W)==length(data.(fields{ii})), ...
                ['cfg.' fields{ii} '.W must have the same number of cells as data.' fields{ii} '.']);
            
            % Reshape W to 2-dims
            for jj = 1:length(cfg.(fields{ii}).W)
                sz = size(cfg.(fields{ii}).W{jj});
                cfg.(fields{ii}).W{jj} = reshape(cfg.(fields{ii}).W{jj},prod(sz(1:end-1)),sz(end));
            end
        case {'beamforming'}
            assert(isfield(cfg.(fields{ii}),'A'), ['Missing cfg.' fields{ii} '.A for bearmforming method.']);
            assert(iscell(cfg.(fields{ii}).A), ['cfg.' fields{ii} '.A must be a cell array.']);
            assert(length(cfg.(fields{ii}).A)==length(data.(fields{ii})), ...
                ['cfg.' fields{ii} '.A must have the same number of cells as data.' fields{ii} '.']);
            if ~isfield(cfg.(fields{ii}),'beamsup')
                cfg.(fields{ii}).beamsup = cell(1,length(data.(fields{ii})));
                for jj = 1:length(data.(fields{ii}))
                    cfg.(fields{ii}).beamsup{jj} = ones(1,length(cfg.(fields{ii}).components{jj}));
                end
            else
                assert(length(cfg.(fields{ii}).beamsup) == length(data.(fields{ii})), ...
                    ['Number of cells in cfg.' fields{ii} '.beamsup does not equal to the number of' ...
                    ' cells in data.' fields{ii} '.']);
                for jj = 1:length(cfg.(fields{ii}).beamsup)
                    assert(length(cfg.(fields{ii}).beamsup{jj}) == length(cfg.(fields{ii}).components{jj}), ...
                        ['Length of cfg.' fields{ii} '.beamsup{' num2str(jj) '} does not equal to' ...
                        ' the number of columns in cfg.' fields{ii} '.components{' num2str(jj) '}.']);
                end
            end
            
            % Reshape W to 2-dims
            for jj = 1:length(cfg.(fields{ii}).A)
                sz = size(cfg.(fields{ii}).A{jj});
                cfg.(fields{ii}).A{jj} = reshape(cfg.(fields{ii}).A{jj},prod(sz(1:end-1)),sz(end));
            end
        otherwise
            error(['Unrecognized cfg.' fields{ii} '.method.']);
    end
    assert(isempty(cfg.(fields{ii}).components) || ...
        (isfield(cfg.(fields{ii}),'components') && length(cfg.(fields{ii}).components) == length(data.(fields{ii}))), ...
        ['cfg.' fields{ii} '.components must have the same number of cells as the data field.']);
    
    % Make specified dimension first
    dimix = find(strcmp(dim.(fields{ii}),cfg.(fields{ii}).dim));
    assert(~isempty(dimix), [cfg.(fields{ii}).dim 'dimension does not exist.']);
    cfgtmp = [];
    cfgtmp.(fields{ii}).shift = dimix-1;
    data = co_shiftdim(cfgtmp,data);
    
    for jj = 1:length(data.(fields{ii}))
        % Make data 2D
        sz = size(data.(fields{ii}){jj});
        data.(fields{ii}){jj} = reshape(data.(fields{ii}){jj}, sz(1), prod(sz(2:end)));
        
        % Determine weights based on method
        switch cfg.(fields{ii}).method
            case 'unmixing'
                assert(size(cfg.(fields{ii}).W{jj},1) == prod(sz(2:end)), ...
                    ['The the first D-1 dimensions in W must equal the product of non-' ...
                    cfg.(fields{ii}).dim ' dimensions.']);
                if isempty(cfg.(fields{ii}).components)
                    comp = 1:size(cfg.(fields{ii}).W,2);
                else
                    comp = cfg.(fields{ii}).components{jj};
                end
                W = cfg.(fields{ii}).W{jj}(:,comp);
            case 'beamforming'
                assert(size(cfg.(fields{ii}).A{jj},1) == prod(sz(2:end)), ...
                    ['The number of rows in A must equal the product of non-' cfg.(fields{ii}).dim ...
                    ' dimensions.']);
                if isempty(cfg.(fields{ii}).components)
                    comp = 1:size(cfg.(fields{ii}).A,2);
                else
                    comp = cfg.(fields{ii}).components{jj};
                end
                W = zeros(prod(sz(2:end)),length(comp));
                beamsup_unq = unique(cfg.(fields{ii}).beamsup{jj});
                
                % Determine which indexes to use to compute the covariance matrix
                if isempty(cfg.(fields{ii}).beamix{jj})
                    beamix = 1:size(data.(fields{ii}){jj},1);
                else
                    beamix = cfg.(fields{ii}).beamix{jj};
                end
                
                if ~isfield(cfg.(fields{ii}),'F')
                    R = data.(fields{ii}){jj}(beamix,:)' * data.(fields{ii}){jj}(beamix,:); % Unfiltered R (chan x chan)
                    R = R./length(beamix);                                                  % Normalize for comparability
                end
                
                for kk = 1:length(beamsup_unq)
                    supix = find(cfg.(fields{ii}).beamsup{jj}==beamsup_unq(kk));    % Indexes of component indexes within suppression group
                    compix = comp(supix);                                           % Component indexes within suppression group
                    
                    if isfield(cfg.(fields{ii}),'F')
                        f1 = cfg.(fields{ii}).F{jj}(:,compix(1));
                        for mm = 2:length(compix)
                            assert(all(cfg.(fields{ii}).F{jj}(:,compix(mm)) == f1), ...
                                'FIR filters within the same suppression group must be identical.');
                        end
                        datatmp = filter(f1,1,data.(fields{ii}){jj},[],1);
                        datatmp = datatmp(beamix,:);
                        R = datatmp' * datatmp;     % chan x chan
                        R = R./size(datatmp,1);     % Normalize for comparability
                    end
                    invR = pinv(R);             % chan x chan
                    
                    if strcmp(cfg.(fields{ii}).beamopt.type,'lcmveig')
                        [V,D] = eig(R);     % Use eigendecomposition of filtered data
                        [D,ix] = sort(diag(D),'descend'); V = V(:,ix); A = pinv(V);
                        V = V(:,1:cfg.(fields{ii}).beamopt.lcmveig.ns); % chan x comp
                        A = A(1:cfg.(fields{ii}).beamopt.lcmveig.ns,:); % comp x chan
                        D = D(1:cfg.(fields{ii}).beamopt.lcmveig.ns); D = diag(D);
                        
                        Rs = V*D*V'; invRs = V*inv(D)*V';
                        invRsA = invRs*cfg.(fields{ii}).A{jj}(:,compix);    % chan x comp
                    end
                    
                    invRA = invR*cfg.(fields{ii}).A{jj}(:,compix);      % chan x comp
                    J = inv(cfg.(fields{ii}).A{jj}(:,compix)'*invRA);   % comp x comp
                    omega = J*invRA'; omega = omega*omega';             % comp x comp
                    
                    if strcmp(cfg.(fields{ii}).beamopt.type,'lcmveig')
                        W(:,supix) = invRsA*J;                          % chan x comp
                        W(:,supix) = W(:,supix)./repmat(sqrt(diag(omega))',size(W,1),1);
                    elseif strcmp(cfg.(fields{ii}).beamopt.type,'bk')
                        % Borgiotti-Kaplan - experimental
                        omega = invRA'*invRA;
                        W(:,supix) = invRA./sqrt(omega);
                    else
                        % lcmv
                        W(:,supix) = invRA*J;                               % chan x comp
                        W(:,supix) = W(:,supix)./repmat(sqrt(diag(omega))',size(W,1),1);
                    end
                end
        end
        
        % Apply dimension reduction
        data.(fields{ii}){jj} = data.(fields{ii}){jj} * W;
        
        % Apply FIR filters
        if isfield(cfg.(fields{ii}),'F')
            F = cfg.(fields{ii}).F{jj}(:,comp);
            for kk = 1:size(data.(fields{ii}){jj},2)
                data.(fields{ii}){jj}(:,kk) = filter(F(:,kk),1,data.(fields{ii}){jj}(:,kk));
            end
        end
    end
    
    % Remove labels from reduced dimensions
    for jj = 1:length(dim.(fields{ii}))
        if ~strcmp(dim.(fields{ii}){jj},cfg.(fields{ii}).dim) && ...
                isfield(data.dim,dim.(fields{ii}){jj}) && ...
                isfield(data.dim.(dim.(fields{ii}){jj}),fields{ii})
            data.dim.(dim.(fields{ii}){jj}) = rmfield(data.dim.(dim.(fields{ii}){jj}),fields{ii});
        end
        
        % Cleanup dimension labels with nothing in them
        if isfield(data.dim,dim.(fields{ii}){jj}) && isempty(data.dim.(dim.(fields{ii}){jj}))
            data.dim = rmfield(data.dim,dim.(fields{ii}){jj});
        end
    end
    
    % Update dimensions
    data.dim.(fields{ii}) = [cfg.(fields{ii}).dim '_comp'];
    
    % Remove events if time dimension was one of the reduced dimensions
    if ~strcmp(cfg.(fields{ii}).dim,'time') && isfield(data.events,fields{ii})
        data.events = rmfield(data.events,fields{ii});
    end
end

% Save cfg settings for future reference
data = co_logcfg(cfg,data);
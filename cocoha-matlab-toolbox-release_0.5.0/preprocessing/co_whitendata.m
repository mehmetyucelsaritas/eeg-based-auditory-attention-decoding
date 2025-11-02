function [data,W] = co_whitendata(cfg,data)
% CO_WHITENDATA performs whitening and unwhitening of specified dimensions. This allows for more
% controlled dimensionality reduction during the whitening process, compared to the bulk whitening
% implemented in CO_COMPUTE_COMPONENTS.
%
% INPUTS:
% cfg.FIELD.dim         = ['time'] variance dimension.
% cfg.FIELD.cell        = [1] data cell to whiten/unwhiten.
%
% cfg.FIELD.operation   = ['compute']/'whiten'/'unwhiten' operation to perform.
%                         'compute' will just compute the whitening matrix structure W without
%                           modifying the data.
%                         'whiten' will perform whitening on the specified data. If cfg.FIELD.W is
%                           specified, its whitening matrix will be used and it will override
%                           cfg.FIELD.dim and cfg.FIELD.whitedim with the parameters provided during
%                           the compute operation. If it is not specified, a whitening matrix
%                           structure will be computed. Whitened dimensions will be collapsed into
%                           the first specified dimension in cfg.FIELD.whitedim, with all dimensions
%                           being turned into singleton dimensions as placeholders. Whitened
%                           dimension labels will also be removed and stored in W. Passing this
%                           updated W structure to the 'unwhiten' operation will restore the labels.
%                         'unwhiten' will perform unwhitening using the cfg.FIELD.W structure.
%                           cfg.FIELD.dim, cfg.FIELD.whitedim and cfg.FIELD.compix will be overriden
%                           based on the parameters provided during 'compute' and 'whiten'
%                           operations.
% cfg.FIELD.selectdim   = {} cell array containing a dimension name - index/label cell pair per
%                         entry. (e.x. {{'chan',1:10}}). Indexes should be specified as an array and
%                         labels should be specified as a cell array. These pairs indicate which
%                         dimensions should be reduced to a limited set of indexes or labels when
%                         computing the whitening matrix. Only used when computing the whitening
%                         matrix.
% cfg.FIELD.whiteix     = {} indexes over cfg.FIELD.dim to be used to compute whitening matrix. Each
%                         numerical array within the cell array specifies the indexes for computing
%                         an individual covariance matrix. All the covariance matrices will then be
%                         averaged together in the end. This is useful when prewhitening for a class
%                         separation procedure in CO_COMPUTECOMPONENTS. If empty, all indexes will
%                         be used.
% cfg.FIELD.whitedim    = ['all']/cell array of dimension(s) to whiten/unwhiten (with exception of
%                         cfg.FIELD.dim).
% cfg.FIELD.compix      = ['all'] whitening matrix component indexes to use for 'whiten' operation.
%
% cfg.FIELD.W           = whitening data structure. Can be optionally specified for the 'whitening'
%                         operation (one will be computed otherwise). Must be specified for the
%                         'unwhiten' operation. Will override cfg.FIELD.dim, cfg.FIELD.whitedim
%                         during the 'whiten' and 'unwhiten' operations, as well as cfg.FIELD.compix
%                         during the 'unwhiten' operation.
%
% OUTPUTS:
% data                  =
% W.
%      FIELD.P          = whitening transformation matrix.
%      FIELD.D          = PCA eigenvalues.
%      FIELD.dim        = variance dimension. Overrides cfg.FIELD.dim.
%      FIELD.whitedim   = dimension names of whitened dimensions. Overrides cfg.FIELD.whitedim.
%      FIELD.whitedimsz = original dimension sizes of whitened dimensions.
%      FIELD.whitedimlb = original labels of whitened dimensions. Only stored after whitening.
%      FIELD.compix     = component indexes used for whitening. Only stored after whitening.
%
%
% See also: CO_COMPUTE_COMPONENTS
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZ
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);

for ii = 1:length(fields)
    if ~isfield(cfg.(fields{ii}),'dim'); cfg.(fields{ii}).dim = 'time'; end;
    if ~isfield(cfg.(fields{ii}),'cell'); cfg.(fields{ii}).cell = 1; end;
    
    if ~isfield(cfg.(fields{ii}),'operation'); cfg.(fields{ii}).operation = 'compute'; end;
    if ~isfield(cfg.(fields{ii}),'whiteix'); cfg.(fields{ii}).whiteix = {}; end;
    if ~isfield(cfg.(fields{ii}),'whitedim'); cfg.(fields{ii}).whitedim = 'all'; end;
    if ~isfield(cfg.(fields{ii}),'selectdim'); cfg.(fields{ii}).selectdim = {}; end;
    for jj = 1:length(cfg.(fields{ii}).selectdim)
        assert(~any(strcmp(cfg.(fields{ii}).selectdim{jj}{1},cfg.(fields{ii}).whitedim)), ...
            ['cfg.' fields{ii} '.selectdim cannot contain the same dimension names as cfg.' ...
            fields{ii} '.whitedim.']);
        assert(~strcmp(cfg.(fields{ii}).selectdim{jj}{1}, cfg.(fields{ii}).dim), ...
            ['cfg.' fields{ii} '.selectdim should not contain the same dimension names as ' ...
            'cfg.' fields{ii} '.dim. See cfg.' fields{ii} '.whiteix.']);
    end
    if ~isfield(cfg.(fields{ii}),'compix'); cfg.(fields{ii}).compix = 'all'; end;
    
    cellix = cfg.(fields{ii}).cell;
    
    % Determine which dimension to compute variance across, and which should be whitenend or not
    if isfield(cfg.(fields{ii}),'W')                                    % cfg.FIELD.W overrides
        dimix = find(strcmp(cfg.(fields{ii}).W.(fields{ii}).dim, ...
            dim.(fields{ii})));                                         % Variance dimension ix
        whitedim = cfg.(fields{ii}).W.(fields{ii}).whitedim;            % Whitening dimension labels
    else
        dimix = find(strcmp(cfg.(fields{ii}).dim,dim.(fields{ii})));
        whitedim = cfg.(fields{ii}).whitedim;
    end
    if ischar(whitedim) && strcmp(whitedim,'all')
        whitedim = dim.(fields{ii}){setdiff(1:length(dim.(fields{ii})),dimix)};
    end
    whitedimix = zeros(1,length(whitedim));                         % Whitening dimension ix
    for jj = 1:length(whitedimix)
        whitedimix(jj) = find(strcmp(whitedim{jj},dim.(fields{ii})));
    end
    %whitedimix = sort(whitedimix);
    nonwhitedimix = setdiff(1:length(dim.(fields{ii})),dimix);      % Non-whitening dimension ix
    nonwhitedimix = setdiff(nonwhitedimix, whitedimix);             % Already in ascending order
    
    % Compute / whiten without cfg.FIELD.W specified
    if strcmp(cfg.(fields{ii}).operation,'compute') || ...
            ( strcmp(cfg.(fields{ii}).operation,'whiten') && ~isfield(cfg.(fields{ii}),'W') )
        % Trim according to cfg.FIELD.selectdim
        datatmp = data;
        for jj = 1:length(cfg.(fields{ii}).selectdim)
            cfgtmp = [];
            cfgtmp.(fields{ii}).cell = cellix;
            cfgtmp.(fields{ii}).dim = cfg.(fields{ii}).selectdim{jj}{1};
            cfgtmp.(fields{ii}).select = cfg.(fields{ii}).selectdim{jj}{2};
            datatmp = co_selectdim(cfgtmp,datatmp);
        end

        % Rearrange datatmp to place dims not to be whitened at very end
        cfgtmp = []; cfgtmp.(fields{ii}).cell = cellix; sz = co_size(cfgtmp,datatmp);
        datatmp = datatmp.(fields{ii}){cellix};
%         sz = size(datatmp);
%         sz = [sz, ones(1,length(dim.(fields{ii}))-length(sz))]; % In case last dim trimmed
        datatmp = permute(datatmp,[dimix, whitedimix, nonwhitedimix]);
        datatmp = reshape(datatmp,[sz(dimix), prod(sz(whitedimix)), prod(sz(nonwhitedimix))]);

        if isempty(cfg.(fields{ii}).whiteix)
            whiteix = {1:size(datatmp,1)};
        else
            whiteix = cfg.(fields{ii}).whiteix;
        end

        % Compute whitening transform
        R = zeros(size(datatmp,2),size(datatmp,2));
        for jj = 1:size(datatmp,3)
            for kk = 1:length(whiteix)
                r = datatmp(whiteix{kk},:,jj)'*datatmp(whiteix{kk},:,jj);
                R = R + r./trace(r);
            end
        end
        R = R./size(datatmp,3)./length(whiteix);
        [V,D] = eig(R); V = real(V); D = real(D); D = diag(D);
        [D,ix] = sort(D,'descend'); V = V(:,ix);
        P = V*diag(sqrt(1./D));
        clear datatmp;
        
        % Create whitening data structure
        W = [];
        W.(fields{ii}).P = P; W.(fields{ii}).D = D;
        W.(fields{ii}).dim = dim.(fields{ii}){dimix};
        W.(fields{ii}).whitedim = whitedim;
        W.(fields{ii}).whitedimsz = sz(whitedimix);
    elseif isfield(cfg.(fields{ii}),'W')
        W = cfg.(fields{ii}).W;
    end
    
    % Whiten data
    if strcmp(cfg.(fields{ii}).operation,'whiten')
        if ischar(cfg.(fields{ii}).compix) && strcmp(cfg.(fields{ii}).compix,'all')
            compix = 1:size(W.(fields{ii}).P,2);
        else
            compix = cfg.(fields{ii}).compix;
        end
        
        cfgtmp = []; cfgtmp.(fields{ii}).cell = cellix; sz = co_size(cfgtmp,data); %size(data.(fields{ii}){cellix});
        data.(fields{ii}){cellix} = permute(data.(fields{ii}){cellix},[dimix, whitedimix, nonwhitedimix]);
        data.(fields{ii}){cellix} = reshape(data.(fields{ii}){cellix}, ...
            [sz(dimix), prod(sz(whitedimix)), prod(sz(nonwhitedimix))]);
        datatmp = zeros(size(data.(fields{ii}){cellix},1), length(compix), ...
            size(data.(fields{ii}){cellix},3));
        for jj = 1:size(data.(fields{ii}){cellix},3)
            datatmp(:,:,jj) = data.(fields{ii}){cellix}(:,:,jj)*W.(fields{ii}).P(:,compix);
        end
        datatmp = reshape(datatmp,[sz(dimix),length(compix), ...
            ones(1,length(whitedimix)-1),prod(sz(nonwhitedimix))]);
        datatmp = ipermute(datatmp,[dimix, whitedimix, nonwhitedimix]);
        data.(fields{ii}){cellix} = datatmp;
        clear datatmp;
        
        % Backup whitening dimension labels and relabel as component numbers
        W.(fields{ii}).whitedimlb = cell(1,length(whitedimix));
        for jj = 1:length(whitedimix)
            if isfield(data.dim,dim.(fields{ii}){whitedimix(jj)}) && ...
                    isfield(data.dim.(dim.(fields{ii}){whitedimix(jj)}),fields{ii})
                W.(fields{ii}).whitedimlb{jj} = ...
                    data.dim.(dim.(fields{ii}){whitedimix(jj)}).(fields{ii}){cellix};
                if jj==1
                    data.dim.(dim.(fields{ii}){whitedimix(jj)}).(fields{ii}){cellix} = ...
                        mat2cell(compix(:)',1,ones(1,length(compix)));
                else
                    data.dim.(dim.(fields{ii}){whitedimix(jj)}).(fields{ii}){cellix} = {1};
                end
            end
        end
        W.(fields{ii}).compix = compix;
    end
    
    % Unwhiten data
    if strcmp(cfg.(fields{ii}).operation,'unwhiten')
        assert(isfield(cfg.(fields{ii}),'W'),'Whitening matrix structure is missing from cfg.');
        assert(isfield(cfg.(fields{ii}).W,fields{ii}),['cfg.' fields{ii} '.W.' fields{ii} ' is absent.']);
        assert(isfield(cfg.(fields{ii}).W.(fields{ii}),'compix'),['Whitening component indexes ' ...
            '(compix) are missing from cfg.' fields{ii} '.W. Please use the W provided by the ' ...
            '''whitening'' operation.']);
        W = cfg.(fields{ii}).W;
        compix = W.(fields{ii}).compix;
        
        sz = size(data.(fields{ii}){cellix});
        data.(fields{ii}){cellix} = permute(data.(fields{ii}){cellix}, ...
            [dimix, whitedimix, nonwhitedimix]);
        data.(fields{ii}){cellix} = reshape(data.(fields{ii}){cellix}, ...
            [sz(dimix), length(compix), prod(sz)/sz(dimix)/length(compix)]);
        datatmp = zeros(size(data.(fields{ii}){cellix},1), size(W.(fields{ii}).P,1), ...
            size(data.(fields{ii}){cellix},3));
        for jj = 1:size(data.(fields{ii}){cellix},3)
            datatmp(:,:,jj) = data.(fields{ii}){cellix}(:,:,jj)*W.(fields{ii}).P(:,compix)';
        end
        datatmp = reshape(datatmp,[sz(dimix),W.(fields{ii}).whitedimsz,sz(nonwhitedimix)]);
        datatmp = ipermute(datatmp,[dimix, whitedimix, nonwhitedimix]);
        data.(fields{ii}){cellix} = datatmp;
        clear datatmp;
        
        % Restore dimension labels
        for jj = 1:length(whitedimix)
            if isfield(data.dim.(dim.(fields{ii}){whitedimix(jj)}),fields{ii})
                if isfield(W.(fields{ii}),'whitedimlb') && ~isempty(W.(fields{ii}).whitedimlb{jj})
                    data.dim.(dim.(fields{ii}){whitedimix(jj)}).(fields{ii}){cellix} = ...
                        W.(fields{ii}).whitedimlb{jj};
                else
                    data.dim.(dim.(fields{ii}){whitedimix(jj)}).(fields{ii}){cellix} = ...
                        mat2cell(1:whitedimsz(jj),1,ones(1,whitedimsz(jj)));
                end
            end
        end
    end
end

% Save cfg settings for future reference
data = co_logcfg(cfg,data);
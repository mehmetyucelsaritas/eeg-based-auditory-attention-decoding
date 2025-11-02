function [n,latent,latentHigh,latentLow] = co_parallelanalysis(cfg,data)
% CO_PATALLELANALYSIS estimates the number of principal components in the data using a Horn's
% Parallel Analysis.
%
% INPUTS:
% cfg.FIELD.cell        = [1]
% cfg.FIELD.dim         = ['time'] observation dimension.
% cfg.FIELD.ix          = ['all'] the indexes in the observation dimension to be used for the analysis.
% cfg.FIELD.nshuffle    = [100] number of random shuffles
% cfg.FIELD.alpha       = [0.05]
%
% Based on Hanan Schteingart's script on the Mathworks File Exchange.
dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);

assert(length(fields)==1, 'Only one field can be specified for this function.');

% Make specified dimension first
dimix = find(strcmp(dim.(fields{1}),cfg.(fields{1}).dim));
assert(~isempty(dimix), [cfg.(fields{1}).dim 'dimension does not exist.']);
cfgtmp = [];
cfgtmp.(fields{1}).shift = dimix-1;
data = co_shiftdim(cfgtmp,data);

% Make data 2D
x = data.(fields{1}){cfg.(fields{1}).cell};
sz = size(x);
x = reshape(x, sz(1), prod(sz(2:end)));
if isnumeric(cfg.(fields{1}).ix)
    x = x(cfg.(fields{1}).ix,:);
end

% [~,~,latent, ~] = pca(x);
[~,D] = eig(x'*x); latent = flipud(diag(D));

xShuffle = x;
latentShuffle = zeros(length(latent), cfg.(fields{1}).nshuffle);
for iShuffle = 1:cfg.(fields{1}).nshuffle
    for dim = 1:size(x,2)
        xShuffle(:, dim) = x(randperm(size(x,1)), dim);
    end
%     [~, ~ ,latentShuffle(:,iShuffle)] = pca(xShuffle);
    [~,D] = eig(xShuffle'*xShuffle); latentShuffle(:,iShuffle) = flipud(diag(D));
end
latentHigh = quantile(latentShuffle', 1-cfg.(fields{1}).alpha);
latentLow = quantile(latentShuffle', cfg.(fields{1}).alpha);

n = sum(latent'>latentHigh);
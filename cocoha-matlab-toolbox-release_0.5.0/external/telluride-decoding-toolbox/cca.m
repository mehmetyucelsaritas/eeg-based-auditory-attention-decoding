function [Wx, Wy, r] = cca(X,Y,regularization,segments)
% CCA calculate canonical correlations
%
% [Wx Wy r] = cca(X,Y) where Wx and Wy contains the canonical correlation
% vectors as columns and r is a vector with corresponding canonical
% correlations. The correlations are sorted in descending order. X and Y
% are matrices where each column is a sample. Hence, X and Y must have
% the same number of columns.
%
% Example: If X is M*K and Y is N*K there are L=MIN(M,N) solutions. Wx is
% then M*L, Wy is N*L and r is L*1.
%
% http://www.imt.liu.se/~magnus/cca/
% (c) 2000 Magnus Borga, Linkopings universitet

% --- Calculate covariance matrices ---
% Modified by malcolm@ieee.org to add regularization parameter
if nargin < 4
    segments = 1;
end
if nargin < 3
    regularization = 10^(-8);
end
if size(X,2) ~= size(Y,2)
    error('Input vectors must have the same number of columns (observations).');
end
if size(X,1) > size(X,2) || size(Y,1) > size(Y,2)
    warning('More dimensions than observations.  Is this right?');
end

% Changed so original vectors are columns, so time goes down (first
% variable)
% X = X';
% Y = Y';

z = [X;Y];
C = covseg(z.',segments); % cov(z.');
sx = size(X,1);
sy = size(Y,1);
Cxx = C(1:sx, 1:sx) + regularization*eye(sx);
Cxy = C(1:sx, sx+1:sx+sy);
Cyx = Cxy';
Cyy = C(sx+1:sx+sy, sx+1:sx+sy) + regularization*eye(sy);
invCyy = inv(Cyy);

% --- Calculate Wx and r ---

[Wx,r] = eig(inv(Cxx)*Cxy*invCyy*Cyx); % Basis in X
r = sqrt(real(r));      % Canonical correlations

% --- Sort correlations ---

V = fliplr(Wx);         % reverse order of eigenvectors
r = flipud(diag(r));    % extract eigenvalues and reverse their order
[r,I]= sort((real(r))); % sort reversed eigenvalues in ascending order
r = flipud(r);          % restore sorted eigenvalues into descending order
for j = 1:length(I)
  Wx(:,j) = V(:,I(j));  % sort reversed eigenvectors in ascending order
end
Wx = fliplr(Wx);        % restore sorted eigenvectors into descending order

% --- Calculate Wy  ---

Wy = invCyy*Cyx*Wx;     % Basis in Y
Wy = Wy./repmat(sqrt(sum(abs(Wy).^2)),sy,1); % Normalize Wy


function C = covseg(x,segs)
x = x-repmat(mean(x,1),size(x,1),1);
seglen = ceil(size(x,1)/segs);
C = zeros(size(x,2));
for ii = 1:segs
    ix1 = (ii-1)*seglen+1;
    ix2 = min(ii*seglen,size(x,1));
    C = C + x(ix1:ix2,:)'*x(ix1:ix2,:);
end
C = C./(size(x,1)-1);
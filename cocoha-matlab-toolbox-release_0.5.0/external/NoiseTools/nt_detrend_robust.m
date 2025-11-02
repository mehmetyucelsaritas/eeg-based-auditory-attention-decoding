function [x,w]=ntdetrend_robust(x,order,w,basis,thresh,niter)
%[y,w]=nt_detrend_robust(x,order,w,basis,thresh,niter) - remove polynomial or sinusoidal trend
% 
%  y: detrended data
%  w: updated weights
%
%  x: raw data
%  order: order of polynomial or number of sin/cosine pairs
%  w: weights
%  basis: 'polynomials' [default] or 'sinusoids', or user-provided matrix
%  thresh: threshold for outliers [default: 2 sd]
%  niter: number of iterations [default: 2]
%
% Noise tools
% See nt_regw().
%
% The data are fit to the basis using weighted least squares. The weight is
% updated by setting samples for which the residual is greater than 'thresh' 
% times its std. to zero, and the fit is repeated at most 'niter'-1 times.
%
% A complex basis (e.g. high order polynomials) might fit the pattern of 
% outliers. To avoid this it may be useful to first detrend using a simpler basis 
% (e.g. linear).
%
% Examples:
% Fit linear trend, ignoring samples > 2*sd from it, and remove:
%   y=nt_detrend_robust(x,1); 
% Fit polynomial trend with initial weighting, threshold = 3*sd
%   y=nt_detrend_robust(x,5,w,3);
% Fit linear then polynomial:
%   [y,w]=nt_detrend_robust(x,1);
%   [yy,ww]=nt_detrend_robust(y,3);
%


%% arguments
if nargin<2; error('!'); end
if nargin<3; w=[]; end
if nargin<4||isempty(basis); basis='polynomials'; end
if nargin<5||isempty(thresh); thresh=2; end
if nargin<6||isempty(niter); niter=2; end

dims=size(x);
x=x(:,:); % concatenates dims >= 2

%% regressors
if isnumeric(basis)
    r=basis;
else
    switch basis
        case 'polynomials'
            r=zeros(size(x,1),numel(order));
            lin=linspace(-1,1,size(x,1));
            for k=1:order
                r(:,k)=lin.^k;
            end
        case 'sinusoids'
            r=zeros(size(x,1),numel(order)*2);
            lin=linspace(-1,1,size(x,1));
            for k=1:order
                r(:,2*k-1)=sin(2*pi*k*lin/2);
                r(:,2*k)=cos(2*pi*k*lin/2);
            end
        otherwise
            error('!');
    end
end
%r=nt_normcol(nt_demean(r));

%% remove trends

% The tricky bit is to ensure that weighted means are removed before
% calculating the regression (see nt_regw).

for iIter=1:niter
    % regress on basis
    [~,y]=nt_regw(x,r,w);
    % residual
    d=x-y;
    % find outliers
    ww=ones(size(x));
    ww(find(abs(d)>thresh*repmat(std(d),size(x,1),1))) = 0;
    % update weights
    if isempty(w); 
        w=ww;
    else
        w=min(w,ww);
    end
end

x=x-y;

%% test code
if 0
    % basic
    x=(1:100)' + randn(size(x));
    y=nt_detrend_robust(x,1);
    figure(1); clf; plot([x,y]);
end
if 0
    % detrend biased random walk
    x=cumsum(randn(1000,1)+0.1);
    y=nt_detrend_robust(x,3,[]);
    figure(1); clf; plot([x,y]); legend('before', 'after');
end
if 0
    % weights
    x=cumsum(rand(100,1));
    x(1:10,:)=100;
    w=ones(size(x)); w(1:10,:)=0;
    y=nt_detrend_robust(x,3,[]);
    yy=nt_detrend_robust(x,3,w);
    figure(1); clf; plot([x,y,yy]); legend('before', 'unweighted','weighted');
end


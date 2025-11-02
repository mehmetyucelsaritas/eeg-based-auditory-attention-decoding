function [g, sparse_g, stats ]=  ADMM_lasso(U,Y,varargin)
%ADMM_lasso: Applies ADMM (alternating direction method of multipliers) to
%            solve lasso problem - l1-regularized LR
%   [g, sparse_g,stats] = lasso(U,Y,...) Performs L1-constrained linear least  
%   squares fits (lasso) relating the stimuli in U to the responses in Y.
%    Solves the following problem via ADMM:
%               minimize 1/2*|| U*g - Y ||_2^2 + \lambda || g ||_1
%   Positional parameters:
%
%     U                A numeric matrix (dimension, say, NxM)
%     Y                A numeric vector of length N
%   
%   Optional input parameters:  
%     'LambdaRatio'    Ratio between the minimum value and maximum value of
%                      lambda to generate, if the  parameter "Lambda" is not 
%                      supplied.  Legal range is (0,1). Default is 0.01.
%
%     'Rho'            The augmented Lagrangian parameter. Default is 1.
% 
%     'Alpha'          The over-relaxation parameter. Typical values for 
%                      alpha are between 1.0 and 1.8. 
%     
%     'AbsTol'         The absolute tolerance. Legal range is (0,Inf). 
%                      Default is 1e-4. 
%     
%     'RelTol'         The relative tolerance. Legal range is (0,Inf).
%                      Default is 1e-2. 
%
%     'MaxIter'        The maxmimum number of iterations. Defaults is 1000.
%
%   Return values:
%     G                The TRF.
%
%     SPARSE_G         The SPARSE TRF.
%
%     STATS            STATS is a struct that contains information about the
%                      model fits. STATS contains the following fields:
%
%       'Loss'         The mean squared error of the fitted model. Loss is
%                      caluclated as:
%                            V = 1/2*|| U*g - Y ||_2^2 + \lambda || sparse_g ||_1
%       'R_Norm'       The sequence of lambda penalties used, in ascending order. 
%                      Dimension 1xL.
%       'S_Norm'       The elastic net mixing value that was used.
%       'eps_pri'      The primal residual.
%       'eps_dual      The dual residual. 
%     
%   References: 
%   [1] Alickovic, R., Lunner, T., Graversen, C., Gustafsson, F., 
%       A sparse estimation approach to modeling listening attention from EEG signals, 
%       submitted to IEEE Transactions on Biomedical Engineering,
%   [2] Boyd, S., Parikh, N., Chu, E., Peleato, B., Eckstein, J., (2010) 
%       Distributed Optimization and Statistical Learning via the Alternating Direction
%       Method of Multipliers, Foundations and Trends in Machine Learning, 
%       Vol. 3, No. 1 (2010) 1–122
%
%
% See also: lasso, CO_DECODE_REGRESSION
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZH
% Author(s): Emina Alickovic


%%%%%%%%%%%%%%%%%%%%%%%% define defaults %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t_start = tic; 
% U a real 2D matrix
if size(Y,1) == 1
    Y = Y';
end

if ~ismatrix(U) || length(size(U)) ~= 2 || ~isreal(U)
    error('The parameter ''U'' must be a real-valued 2D matrix.');
end

if size(U,1) < 2
    error('The parameter ''U'' must have at least two rows.');
end

% Y has same number of columns as U
if ~isreal(Y) || size(U,1) ~= size(Y,1)
    error('The parameter ''Y'' must be a real-valued vector/matrix with ... column dimension same as the columns of ''U''.');
end

pnames = {'lambdaratio' 'rho' 'alpha' 'abstol' 'reltol' 'maxiter'};
dflts  = { 0.01          1      1       1e-4     1e-2   1000};
[lambdarat, rho, alpha, abstol, reltol, maxiter]= internal.stats.parseArgs(pnames, dflts, varargin{:});
if lambdarat >=1
    lambdarat = 0.01;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% data preprocessing %%%%%%%%%%%%%%%%%%%%


U = normalizeMatrix(U); 
Y = normalizeMatrix(Y);

% cache U'Y
UtY = U'*Y;

% compute lambda 
dotp = abs(UtY); lambda_max = max(dotp);
lambda =  lambdarat*lambda_max;

[j, k] = size(U);
g = zeros(k,1);  
sparse_g = zeros(k,1); 
delta = zeros(k,1); 

% cache the factorization: U'U + rho*I
[L1, U1] = cacheFactor(U, rho);

%%%%%%%%%%%%%%%%%%%%%%% solving lasso problem with ADMM %%%%%%%%%%%%%%%
for ii = 1 : maxiter
    
	% g update
    uu = UtY + rho*(sparse_g - delta);    
    if j >= k 
        g = U1 \ (L1 \ uu); 
    else
        % if the matrix is fat, we can use matrix inversion to factor the
        % smaller matrxi U*U' + rho*I instead
        g = uu/rho - (U'*(U1 \ ( L1 \ (U*uu) )))/rho^2;
    end
    
    % g_sparse update'
    g_sparse_init = sparse_g;
    g_est = alpha*g + (1-alpha)* g_sparse_init;
    sparse_g = soft_tresholding(g_est + delta, lambda/rho);
    
    
    % dual variable (delta) update
    delta = delta + (g_est - sparse_g);
    
    % compute statistics
    stats.loss(ii) = loss_function(U, Y, lambda, g, sparse_g);
    
    stats.r_norm(ii) = sqrt(normL2(g-sparse_g));
    gsdiff = sparse_g - g_sparse_init;
    stats.s_norm(ii) = sqrt(-rho*normL2(gsdiff));
    
    stats.eps_pri(ii) = sqrt(ii)*abstol + reltol*max(normL2(g), normL2(-sparse_g));
    stats.eps_dual(ii)= sqrt(ii)*abstol + reltol*normL2(rho*delta);
    
    if (stats.r_norm(ii) < stats.eps_pri(ii) && ...
       stats.s_norm(ii) < stats.eps_dual(ii))
         break;
    end
end

toc(t_start);

end


%%%%%%%%%%%%%%%%%%%%% helper functions %%%%%%%%%%%%%%%%
function X_sparse = normalizeMatrix(X)
n = size(X,2);
X_sparse = X*spdiags(1./sqrt(sum(X.^2))',0,n,n); 
end 

function [L, U] = cacheFactor(U, rho)
[u1, u2] = size(U); 
if (u1>u2) 
    L = chol( U'*U + rho*speye(u2), 'lower' );
else
    L = chol(1/rho*(U*U') + speye(u1), 'lower' );
end
 L = sparse(L);
    U = sparse(L');
end

function s = soft_tresholding(a, kappa)
   s = max( 0, a - kappa ) - max( 0, -a - kappa );
end

function L = loss_function(U, Y, lambda, g, g_sparse)
g1 = sum(abs(g_sparse));
L = 1/2*sum((Y-U*g).^2) + lambda*g1;
end

function l2 = normL2(w)
l2 = sqrt(w'*w);
end
function z=nt_multiscale(x,depth)
%z=nt_multiscale(x,depth) - apply smoothing at multiple scales
%
%   y: result
%
%   x: data
%   depth: depth of scales to investigate
% 
% NoiseTools

if nargin<2; error('!'); end

if ndims(x)==3;
    [m,n,o]=size(x);
    z=zeros(m-2^depth-1,n*depth,o);
    for k=1:o
        z(:,:,k)=nt_multiscale(x(:,:,k),depth);
    end
    return
end
    
[m,n]=size(x);
z=zeros(m,n,depth);

z(:,:,1)=x(1:size(z,1),:);
for k=1:depth-1
    step=2^k-1;
    idx=1:(m-step);
    z(idx,:,k+1) = (...
        z(idx,:,k) + ...
        z(idx+step,:,k) )/2;
end

z=reshape(z,[m,n*depth]);
z=z(1:end-2^depth-1,:);
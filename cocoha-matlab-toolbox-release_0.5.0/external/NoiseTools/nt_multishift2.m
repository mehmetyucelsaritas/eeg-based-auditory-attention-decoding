function z=nt_multishift2(x,nshifts)
%z=nt_multishift2(x,nshifts) - apply multiple shifts/smoothing to matrix
%
%   y: result
%
%   x: matrix to shift
%   nshifts: number of shift terms
% 
% X is shifted column by column (all shifts of 1st column, then all
% shifts of second column, etc).
% 
% X may be 1D, 2D or 3D. See also convmtx.
%
% NoiseTools


TBD


if size(x,1)<max(shifts); error('shifts should be no larger than nrows'); end
if min(shifts)<0; error('shifts should be nonnegative'); end
shifts=shifts(:)';
nshifts=numel(shifts);

% array of shift indices
N=size(x,1)-max(shifts); 
shiftarray=nt_vecadd(nt_vecmult(ones(N,nshifts),shifts),(1:N)');
[m,n,o]=size(x);
z=zeros(N,n*nshifts,o);

if ~isempty(amplitudes)
    amplitudes=amplitudes(:)';
    if numel(amplitudes)~=numel(shifts); error('amplitudes and shifts should have same numel'); end
    for k=1:o
        for j=0:n-1
            y=x(:,j+1,k);
            z(:,j*nshifts+1: j*nshifts+nshifts,k)=nt_vecmult(y(shiftarray),amplitudes);
        end
    end
else
    for k=1:o
        for j=0:n-1
            y=x(:,j+1,k);
            z(:,j*nshifts+1: j*nshifts+nshifts,k)=y(shiftarray);
        end
    end
end


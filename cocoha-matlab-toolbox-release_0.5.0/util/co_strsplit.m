function C = co_strsplit(s, delimiter)
% CO_STRSPLIT allows backward compatibility with MATLAB versions that predate the STRSPLIT function.
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU/Oticon, UCL, ETH Zurich
% Author(s): Daniel D.E. Wong

if ~exist('strsplit','file')
    C = strread(s,'%s','delimiter',delimiter);
else
    C = strsplit(s,delimiter);
end
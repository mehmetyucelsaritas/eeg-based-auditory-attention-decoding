function pyArray = co_mat2numpy(mat,floatX)
% CO_MAT2NUMPY converts a Matlab matrix into a NumPy array with the same dimensions.
%
% INPUTS:
% mat       = the Matlab matrix.
% floatX    = [false] whether to use floatX as defined in theano.config, provided Theano is
%             installed.
%
% OUTPUTS:
% pyArray   = the converted NumPy array.
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU/Oticon, UCL, ETH Zurich
% Author(s): Daniel D.E. Wong

if ~exist('floatX','var'); floatX = false; end;
assert(~isempty(which('py')),'Matlab version must support Python calls (2014b or higher)');
try py.imp.find_module('numpy'); catch; error('NumPy module not found.'); end;

args = {};
if floatX
    try
        py.imp.find_module('theano');
        theanocfg = py.theano.config;
        args = {'dtype',theanocfg.floatX};
    catch
    end
end

sz = size(mat);
dimix = 1:length(sz); dimix(1) = 2; dimix(2) = 1; mat = permute(mat,dimix);
pyArray = py.numpy.array(mat(:)',pyargs(args{:})); %varargin{:}));
pyArray = pyArray.reshape(int32(sz));

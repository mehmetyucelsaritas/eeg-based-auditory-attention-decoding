function mat = co_numpy2mat(pyArray)
% CO_NUMPY2MAT converts a NumPy ND-array into a Matlab array.
%
% INPUTS:
% pyArray       = the NumPy array.
%
% OUTPUTS:
% mat           = the Matlab matrix.
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZ
% Author(s): Daniel D.E. Wong

assert(~isempty(which('py')),'Matlab version must support Python calls (2014b or higher)');
try py.imp.find_module('numpy'); catch; error('NumPy module not found.'); end;

pyType = py.type(pyArray);
assert(strcmp(char(py.eval('pyType.__module__ + "." + pyType.__name__',py.dict(pyargs('pyType',pyType)))), 'numpy.ndarray'), ...
    'Input is not of type numpy.ndarray');

sz = double(py.array.array('d',py.numpy.nditer(py.numpy.array(pyArray.shape))));
if strcmp(char(py.eval('pyType.__module__ + "." + pyType.__name__',py.dict(pyargs('pyType',pyArray.dtype.type)))), 'numpy.string_')
    mat = cell(py.list(py.numpy.nditer(pyArray)));
    for ii = 1:numel(mat); mat{ii} = char(mat{ii}); end;
else
    mat = double(py.array.array('d',py.numpy.nditer(pyArray)));
end

if length(sz) == 1; sz = [sz 1]; end;   % Default to column vector
sz_mat = sz; sz_mat(1) = sz(2); sz_mat(2) = sz(1);
mat = reshape(mat,sz_mat);

dimix = 1:length(sz); dimix(1) = 2; dimix(2) = 1;
mat = permute(mat,dimix);
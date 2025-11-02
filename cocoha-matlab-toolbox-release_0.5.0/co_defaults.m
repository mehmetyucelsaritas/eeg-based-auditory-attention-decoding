function co_defaults
% CO_DEFAULTS sets the MATLAB paths for using the COCOHA toolbox.
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZ
% Author(s): Daniel D.E. Wong

addpath(fileparts(which('co_defaults')));
addpath(fullfile(fileparts(which('co_defaults')), 'decoders'));
addpath(fullfile(fileparts(which('co_defaults')), 'models'));
addpath(fullfile(fileparts(which('co_defaults')), 'models', 'cochlea'));
addpath(fullfile(fileparts(which('co_defaults')), 'pipeline'));
addpath(fullfile(fileparts(which('co_defaults')), 'preprocessing'));
addpath(fullfile(fileparts(which('co_defaults')), 'util'));
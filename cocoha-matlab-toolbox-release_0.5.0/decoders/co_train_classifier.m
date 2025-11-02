function [decoder,train_ix,val_ix] = co_train_classifier(cfg,data)
% CO_TRAIN_CLASSIFIER trains an classifier on the data. One dimension is expected to contain the
% class in its dimension labels. All other dimensions will be collapsed and used as features. For
% Matlab-based classifiers, the Statistics & Machine Learning and Neural Network Matlab toolboxes
% are supported. For Python-based classifiers, the SciKit-Learn and Keras Python packages are 
% supported.
%
% INPUTS:
% * Note that there should only be one cfg.FIELD.
% cfg.FIELD.dim         = dimension containing class labels in dimension values.
% cfg.FIELD.cells       = ['all'] an array listing the cells from which features are taken.
% cfg.FIELD.train       = [1] indexes used for training data or fraction of data used for training.
%                         If a fraction is specified, the fraction will be drawn from each data
%                         class to ensure all classes are represented.
% cfg.FIELD.validation  = indexes used for validation data or fraction of data used for validation
%                         If a fraction is specified, the fraction will be drawn from each data
%                         class to ensure all classes are represented. A fraction can only be 
%                         specified if this has also been done for cfg.FIELD.train. A validation set
%                         is only needed for DNN training.
% cfg.FIELD.zscore      = ['yes']/'no' scales training data to zero-mean and unit variance. All
%                         other data passed to the classifier will be scaled using the same
%                         parameters.
% cfg.FIELD.classalgo   = 'svm'/'pysvm'/'pysvmlin'/'nb'/'pynb'/'knn'/'pyknn'/'rf'/'pyrf'/'dnn'/
%                         'pydnn' classification algorithm to use. Python implementations denoted by
%                         py* prefix.
%                         'svm' trains a multiway SVM classifier using a one-vs-one scheme.
%                         'svmlin' trains a linear multiway one-vs-rest SVM classifier.
%                         'nb' trains a Naive Bayes classifier.
%                         'knn' trains a K-Nearest Neighbor classifier.
%                         'rf' trains a random forest classifier.
%                         'dnn' trains a DNN layer by layer using minibatch training. Automatically
%                           balances training and validation sets.
%                         'pysvm' trains a multiway SVM classifier using Python SciKit-Learn.
%                         'pynb' trains a Gaussian Naive Bayes classifier using Python SciKit-Learn.
%                           This is the only Naive Bayes class in SciKit-Learn that handles negative
%                           feature values.
%                         'pyknn' trains a K-Nearest Neighbor classifier using Python SciKit-Learn.
%                         'pyrf' trains a random forest classifier using Python SciKit-Learn.
%                         'pydnn' trains a DNN layer by layer using the Keras package in Python.
%
% cfg.FIELD.svm.opt     = {} cell array containing name-value pairs for FITCECOC.
% cfg.FIELD.svm.svmopt  = {} cell array containing name-value pairs for TEMPLATESVM.
%
% cfg.FIELD.nb.opt      = {} cell array containing name-value pairs for FITCNB.
%
% cfg.FIELD.knn.opt     = {} cell array containing name-value pairs for FITCKNN.
%
% cfg.FIELD.rf
% cfg.FIELD.rf  .ntree  = [500] number of trees.
%               .mtry   = [] default is the square root of the number of features.
%               .opt    = {} cell array containing name-value pairs for CLASSRF_TRAIN.
%
% cfg.FIELD.dnn
%               .layers         = array of layer sizes for hidden layers.
%               .activation     = (optional) cell array indicating activation type of each layer
%                                 (e.g. 'logsig'/'poslin'/other Matlab Neural Network Toolbox
%                                 transfer function). Default is to use logsig for all layers.
%               .rbmEpochs      = array of number of epochs of RBM pretraining for each added
%                                 layer. 0 indicates no RBM pretraining. By default this is 0
%                                 for each layer.
%               .rbmType        = (optional) cell array indicating type of RBM for each layer
%                                 (e.g. {'generative','discriminative'} for a 2 layer network.
%                                 Default is for a generative network with a discriminative
%                                 final hidden layer.
%               .trainfcn       = ['trainscg']/'traingdx' training algorithm to use.
%               .freeze         = ['no']/'yes' whether to prevent previously trained layers from
%                                 being trained along with newly appended layer.
%               .nBatch         = number of batches.
%               .gwn            = [0] standard deviation of Gaussian white noise added to
%                                 training inputs.
%               .nTerm          = [100] number of training epochs over which a lack of
%                                 performance improvement will result in termination of training.
%                                 Can be specified as a single number for all layers, or one per
%                                 layer.
%               .maxEpoch       = [1000] maximum number of training epochs. Can be specified as a
%                                 single number for all layers, or one per layer.
%               .minEpoch       = [0] minimum number of epochs before best network is selected
%                                 and epoch restarting with cfg.minVal takes effect. Can be
%                                 specified as a single number for all layers or one per layer.
%               .checkEpoch     = [5] number of epochs after which validation score will be
%                                 reported (although score is computed every epoch). Can be
%                                 specified as a single number for all layers, or one per layer.
%               .nTries         = [1] number of training attempts per layer. Network with best
%                                 validation result will be chosen. Can be specified as a single
%                                 number for all layers, or one per layer.
%               .minVal         = [-Inf] minimum score that must be obtained by cfg.minEpoch.
%                                 Otherwise, network will be rejected and the epoch will be
%                                 restarted. Training run will not count against cfg.nTries. Can be
%                                 specified as a single number for all layers, or one per layer.
%               .valType        = ['negloss']/'score'/'area' method used to determine validation
%                                 score reported at cfg.checkEpoch and used by cfg.minVal.
%               .keepType       = ['negloss']/'area'/'score' method used to determine which network
%                                 to select.
%               .checkCurve     = ['trainEnd']/'checkEpoch'/'no' if classification curve is not
%                                 increasing, network will be discarded. 'trainEnd' only
%                                 evaluates at the end of training the layer. 'checkEpoch' will
%                                 check every cfg.checkEpoch for epochs > cfg.minEpoch. The
%                                 training run will not count against cfg.nTries.
%               .truncate       = ['no'] truncates network at previous layer if additional
%                                 layer doesn't improve performance as measured by cfg.keepType.
%               .gpu            = ['no']/'yes' whether to use the GPU.
%
% cfg.FIELD.pysvm.opt           = {} cell array containing name-value pairs for <a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">sklearn.svm.SVC</a>.
%
% cfg.FIELD.pysvmlin.opt        = {} cell array containing name-value pairs for <a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">sklearn.svm.SVC</a>.
%
% cfg.FIELD.pynb.opt            = {} cell array containing name-value pairs for <a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html"sklearn.svm.LinearSVC</a>.
%                                 <a href="http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html">sklearn.naive_bayes.GaussianNB</a>.
%
% cfg.FIELD.pyknn.opt           = {} cell array containing name-value pairs for
%                                 <a href="http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">sklearn.neighbors.KNeighborsClassifier</a>.
%
% cfg.FIELD.pyrf.opt            = {} cell array containing name-value pairs for
%                                 <a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">sklearn.ensemble.RandomForestClassifier</a>.
%
% cfg.FIELD.pydnn               = same options as cfg.FIELD.dnn with the following exceptions:
%                   .trainfcn   = ['sgd'] See http://keras.io/optimizers/ for other options.
%                   .activation = (optional) cell array indicating activation type of each layer
%                                 (e.g. 'sigmoid'/'relu'/<a href="http://keras.io/activations/"see Keras documentation</a>). The
%                                 default is to sigmoid for all layers.
%                   .l1         = [0] L1 regularization parameter. Can also be specified per layer.
%                   .l2         = [0] L2 regularization parameter. Can also be specified per layer.
%                   .pyEpochs   = [1] number of epochs to run on Keras before returning 1 epoch in
%                                 Matlab. Epochs in Keras are faster, but validation scores in
%                                 Matlab will not be updated. Can be specified as a single number
%                                 for all layers or one per layer.
%                   .gpu        = ['no']/'yes' whether to use the GPU for RBM pretraining. GPU usage
%                                 for actual training depends on Theano's configuration. See
%                                 http://deeplearning.net/software/theano/install.html#using-the-gpu
%                                 Note if both DeeBNet (RBM) and Theano (DNN) are set to use the
%                                 GPU, a resource conflict may arise.
%                   .theanoL    = ['c|py_nogc'] Theano <a href="http://deeplearning.net/software/theano/tutorial/modes.html#linkers">linker</a>.
%
% data                          = data structure containing features for classification.
%
%
% OUTPUTS:
% decoder                   = CoDecoder instance with the following fields:
%           .type           = 'mat_class' identifier of type of decoder.
%           .dim            = dimension containing class labels in dimension values.
%           .spec.classalgo = classification algorithm used.
%           .spec.nfeat     = number of features decoder should expect.
%           .spec.onehotmap = one-hot encoding mapping (used only for DNN).
%           .spec.scale     = z-score scaling coefficients (means and standard deviations) for input
%                             data.
%           .decoder        = the SVM or DNN decoder.
%           .cfg            = decoder configuration structure.
%
% train_ix                  = indexes along the class label dimension corresponding to data used for
%                             training.
% val_ix                    = indexes along the class label dimension corresponding to data used for
%                             validation.
%
% See also: CO_DECODE_CLASSIFIER
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZ
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);

% Asserts
assert(length(fields)==1,'There should only be one cfg.FIELD.');
assert(isfield(cfg.(fields{1}),'classalgo'),'No classification algorithm specified.');
assert(~strcmp(cfg.(fields{1}),'dnnlayerwise') || isfield(cfg.(fields{1}),'validation'), ...
    'Validation set must be specified for DNN classifier.');

% Set defaults
if ~isfield(cfg.(fields{1}),'dim'); cfg.(fields{1}).dim = 'time'; end;
if ~isfield(cfg.(fields{1}),'cells'); cfg.(fields{1}).cells = 'all'; end;
if ~isfield(cfg.(fields{1}),'train'); cfg.(fields{1}).train = 1; end;
if ~isfield(cfg.(fields{1}),'zscore'); cfg.(fields{1}).zscore = 'yes'; end;
if ~isfield(cfg.(fields{1}),'svm') || ~isfield(cfg.(fields{1}).svm,'opt')
    cfg.(fields{1}).svm.opt = {};
end
if ~isfield(cfg.(fields{1}).svm,'svmopt'); cfg.(fields{1}).svm.svmopt = {}; end;
if ~isfield(cfg.(fields{1}),'nb') || ~isfield(cfg.(fields{1}).nb,'opt')
    cfg.(fields{1}).nb.opt = {};
end
if ~isfield(cfg.(fields{1}),'knn') || ~isfield(cfg.(fields{1}).knn,'opt')
    cfg.(fields{1}).knn.opt = {};
end
if ~isfield(cfg.(fields{1}),'rf'); cfg.(fields{1}).rf = []; end
if ~isfield(cfg.(fields{1}).rf,'ntree'); cfg.(fields{1}).rf.ntree = 500; end;
if ~isfield(cfg.(fields{1}).rf,'mtry'); cfg.(fields{1}).rf.mtry = 500; end;
if ~isfield(cfg.(fields{1}).rf,'opt'); cfg.(fields{1}).rf.opt = {}; end;
if ~isfield(cfg.(fields{1}),'dnn'); cfg.(fields{1}).dnn = []; end;
if ~isfield(cfg.(fields{1}),'pylogit') || ~isfield(cfg.(fields{1}).pylogit,'opt')
    cfg.(fields{1}).pylogit.opt = {};
end
if ~isfield(cfg.(fields{1}),'pysvm') || ~isfield(cfg.(fields{1}).pysvm,'opt')
    cfg.(fields{1}).pysvm.opt = {};
end
if ~isfield(cfg.(fields{1}),'pysvmlin') || ~isfield(cfg.(fields{1}).pysvmlin,'opt')
    cfg.(fields{1}).pysvmlin.opt = {};
end
if ~isfield(cfg.(fields{1}),'pynb') || ~isfield(cfg.(fields{1}).pynb,'opt')
    cfg.(fields{1}).pynb.opt = {};
end
if ~isfield(cfg.(fields{1}),'pyknn') || ~isfield(cfg.(fields{1}).pyknn,'opt')
    cfg.(fields{1}).pyknn.opt = {};
end
if ~isfield(cfg.(fields{1}),'pyrf') || ~isfield(cfg.(fields{1}).pyrf,'opt')
    cfg.(fields{1}).pyrf.opt = {};
end

% Initialize decoder structure
decoder = CoDecoder();
decoder.type = 'class';
decoder.dim = cfg.(fields{1}).dim;
decoder.spec.classalgo = cfg.(fields{1}).classalgo;

% Convert data to classification features
cfgtmp = [];
cfgtmp.(fields{1}).dim = cfg.(fields{1}).dim;
if ischar(cfg.(fields{1}).cells) && strcmp(cfg.(fields{1}).cells,'all')
    cfgtmp.(fields{1}).cells = 1:length(data.(fields{1}));
else
    cfgtmp.(fields{1}).cells = cfg.(fields{1}).cells;
end
cfgtmp.(fields{1}).onehot = 'yes';   % Use one-hot encoding for training/validation fractions
[feat_x, feat_y, onehotmap] = co_data2class(cfgtmp, data);
decoder.spec.nfeat = size(feat_x,2);

% Create training set
if length(cfg.(fields{1}).train) == 1
    assert(cfg.(fields{1}).train <= 1,'Training fraction should not exceed 1.');
    
    classix = cell(1,length(onehotmap));
    train_ix = [];
    for ii = 1:length(onehotmap)
        classix{ii} = find(feat_y(:,ii));
        train_ix = [train_ix; classix{ii}(1:round(length(classix{ii})*cfg.(fields{1}).train))];
    end
else
    train_ix = cfg.(fields{1}).train;
end

% If not DNN or RandomForest, convert one-hot back to classes
if ~strcmp(cfg.(fields{1}).classalgo,'dnn') && ...
        ~strcmp(cfg.(fields{1}).classalgo,'rf') && ~strcmp(cfg.(fields{1}).classalgo,'pydnn')
    if iscell(onehotmap)
        feat_y_tmp = cell(size(feat_y,1),1);
        for ii = 1:size(feat_y,1); feat_y_tmp{ii} = onehotmap{logical(feat_y(ii,:))}; end;
    else
        feat_y_tmp = zeros(size(feat_y,1),1);
        for ii = 1:size(feat_y,1); feat_y_tmp(ii) = onehotmap(logical(feat_y(ii,:))); end;
    end
    feat_y = feat_y_tmp;
else
    decoder.spec.onehotmap = onehotmap;
end

% Create validation set
if isfield(cfg.(fields{1}),'validation')
    if length(cfg.(fields{1}).validation) == 1
        assert(length(cfg.(fields{1}).train) == 1,...
            ['Can only specify cfg.FIELD.validation as a fraction when cfg.FIELD.train ' ...
            'is also a fraction.']);
        assert(cfg.(fields{1}).train+cfg.(fields{1}).validation <= 1, ...
            'Validation fraction extends beyond dataset.');
        val_ix = [];
        for ii = 1:length(onehotmap)
            val_ix = [val_ix; classix{ii}(round(length(classix{ii})*cfg.(fields{1}).train)+1: ...
                round(length(classix{ii})*(cfg.(fields{1}).train+cfg.(fields{1}).validation)))];
        end
    else isfield(cfg.(fields{1}),'validation')
        val_ix = cfg.(fields{1}).validation;
    end
elseif strcmp(cfg.(fields{1}).classalgo,'dnn') || strcmp(cfg.(fields{1}).classalgo,'pydnn')
    % DNN needs a validation set - use training data if none specified
    val_ix = train_ix;
else
    val_ix = [];
end
train_x = feat_x(train_ix,:);
train_y = feat_y(train_ix,:);
val_x = feat_x(val_ix,:);
val_y = feat_y(val_ix,:);

% Compute and apply data scaling coefficients
if strcmp(cfg.(fields{1}).zscore,'yes')
    trainx_mean = mean(train_x,1);
    trainx_std = std(train_x,[],1);
    decoder.spec.scale.mean = trainx_mean;
    decoder.spec.scale.std = trainx_std;
    
    train_x = (train_x-repmat(trainx_mean,size(train_x,1),1))./repmat(trainx_std,size(train_x,1),1);
    val_x = (val_x-repmat(trainx_mean,size(val_x,1),1))./repmat(trainx_std,size(val_x,1),1);
else
    decoder.spec.scale.mean = zeros(1,size(train_x,2));
    decoder.spec.scale.std = ones(1,size(train_x,2));
end

% Train classifier
switch cfg.(fields{1}).classalgo
    case 'svm'
        assert(exist('fitcecoc','file')==2, ...
            'The Matlab Statistics and Machine Learning Toolbox is needed for SVM.');
        t = templateSVM('SaveSupportVectors','on',cfg.(fields{1}).svm.svmopt{:});
        decoder.decoder = compact(fitcecoc(train_x,train_y,'Learners',t, ...
            cfg.(fields{1}).svm.opt{:}));
    case 'nb'
        assert(exist('fitcnb','file')==2, ...
            'The Matlab Statistics and Machine Learning Toolbox is needed for Naive Bayes.');
        decoder.decoder = fitcnb(train_x,train_y,cfg.(fields{1}).nb.opt{:});
    case 'knn'
        assert(exist('fitcknn','file')==2, ...
            'The Matlab Statistics and Machine Learning Toolbox is needed for K-Nearest Neigbor.');
        decoder.decoder = fitcknn(train_x,train_y,cfg.(fields{1}).knn.opt{:});
    case 'rf'
        assert(exist('classRF_train','file')==2, ...
            ['Please add RandomForest-Matlab package ' ...
            '(https://github.com/jrderuiter/randomforest-matlab) RF_Class_C folder to the ' ...
            'Matlab path and compile it.']);
        
        % classRF_train doesn't handle cell labels - Convert from one-hot to indexes
        [~,train_y] = max(train_y,[],2);
        
        decoder.decoder = classRF_train(train_x,train_y,cfg.(fields{1}).rf.ntree, ...
            cfg.(fields{1}).rf.mtry, cfg.(fields{1}).rf.opt{:});
    case 'dnn'
        [decoder.decoder,~,cfg.(fields{1}).dnnlayerwise] = train_layerwise( ...
            cfg.(fields{1}).dnn, ...
            train_x',train_y',val_x',val_y');
    case 'pylogit'
        checkPyModule('sklearn');
        decoder.decoder = py.sklearn.linear_model.LogisticRegression( ...
            pyargs(cfg.(fields{1}).pylogit.opt{:}));
        train_x_np = co_mat2numpy(train_x); train_y_np = co_mat2numpy(train_y);
        decoder.decoder.fit(train_x_np,train_y_np);
    case 'pysvm'
        checkPyModule('sklearn');
        decoder.decoder = py.sklearn.svm.SVC(pyargs(cfg.(fields{1}).pysvm.opt{:}));
        train_x_np = co_mat2numpy(train_x); train_y_np = co_mat2numpy(train_y);
        decoder.decoder.fit(train_x_np,train_y_np);
    case 'pysvmlin'
        checkPyModule('sklearn');
        decoder.decoder = py.sklearn.svm.LinearSVC(pyargs(cfg.(fields{1}).pysvmlin.opt{:}));
        train_x_np = co_mat2numpy(train_x); train_y_np = co_mat2numpy(train_y);
        decoder.decoder.fit(train_x_np,train_y_np);
    case 'pynb'
        checkPyModule('sklearn');
        decoder.decoder = py.sklearn.naive_bayes.GaussianNB(pyargs(cfg.(fields{1}).pynb.opt{:}));
        train_x_np = co_mat2numpy(train_x); train_y_np = co_mat2numpy(train_y);
        decoder.decoder.fit(train_x_np,train_y_np);
    case 'pyknn'
        checkPyModule('sklearn');
        decoder.decoder = py.sklearn.neighbors.KNeighborsClassifier(pyargs( ...
            cfg.(fields{1}).pyknn.opt{:}));
        train_x_np = co_mat2numpy(train_x); train_y_np = co_mat2numpy(train_y);
        decoder.decoder.fit(train_x_np,train_y_np);
    case 'pyrf'
        checkPyModule('sklearn');
        decoder.decoder = py.sklearn.ensemble.RandomForestClassifier(pyargs( ...
            cfg.(fields{1}).pyrf.opt{:}));
        train_x_np = co_mat2numpy(train_x); train_y_np = co_mat2numpy(train_y);
        decoder.decoder.fit(train_x_np,train_y_np);
    case 'pydnn'
        [decoder.decoder,~,cfg.(fields{1}).dnnlayerwise] = train_pylayerwise( ...
            cfg.(fields{1}).pydnn, ...
            train_x',train_y',val_x',val_y');
    otherwise
        error('Unrecognized classification algorithm specified.');
end

% Save cfg settings for future reference
[stck,stckix] = dbstack;
cfg.fcn = stck(stckix).name;
cfg.date = date;
cfg.datacfg = data.cfg;
decoder.cfg{1} = cfg;


function checkPyModule(module)
try py.imp.find_module(module);
catch; error(['The Python ' module ' module was not found.']); end;
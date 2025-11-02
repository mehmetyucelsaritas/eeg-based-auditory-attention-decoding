function [finalnet,valscore,cfg] = train_pylayerwise(cfg,train_x,train_y,val_x,val_y)
% TRAIN_PYLAYERWISE trains a DNN layer by layer using minibatch training. Restricted Boltzmann
% Machine pretraining is available via the DeeBNet toolbox. The DNN itself uses the Python Keras
% package. Class balancing for training and validation sets is automatically performed.
%
% Theano Setup Notes:
% 1) PATH environment variable should include the CUDA bin directory (where nvcc is located). This
%    is typically /usr/local/cuda/bin on Linux systems.
% 2) CUDA_ROOT environment variable should be set to the CUDA root directory. This is typically
%    /usr/local/cuda on Linux systems.
% 3) LD_LIBRARY_PATH environment variable should include the CUDA lib directory. This is typically 
%    /usr/local/cuda/lib64 on 64-bit Linux systems.
% 4) THEANO_FLAGS environment variable should be set to use float32. GPU/CPU usage can also be set
%    here i.e. THEANO_FLAGS='floatX=float32,device=gpu,mode=FAST_RUN'.
%
% INPUTS:
% cfg.layers    = array of layer sizes.
% cfg.activation= (optional) cell array indicating activation type of each layer (e.g. 'sigmoid'/
%                 'relu'/<a href="http://keras.io/activations/"see Keras documentation</a>). The
%                 default is to sigmoid for all layers.
% cfg.l1        = [0] L1 regularization parameter. Can be specified per layer.
% cfg.l2        = [0] L2 regularization parameter. Can be specified per layer.
% cfg.rbmEpochs = [0] number of epochs of RBM pretraining for each added layer. 0 indicates no
%                 RBM pretraining. Can be specified as a single number for all layers, or one per
%                 layer.
% cfg.rbmType   = (optional) cell array indicating type of RBM for each layer (e.g. {'generative',
%                 'discriminative'} for a 2 layer network. Default is for a generative network with
%                 a discriminative final hidden layer.
% cfg.trainfcn  = ['sgd']/'rmsprop'/'adagrad'/'adadelta'/'adam'/'adamax' training algorithm to use.
%                 See http://keras.io/optimizers/.
% cfg.freeze =    ['no']/'yes' whether to prevent previously trained layers from being trained along
%                 with newly appended layer.
% cfg.nBatch    = number of batches.
% cfg.gwn       = [0] standard deviation of Gaussian white noise added to training inputs.
% cfg.nTerm     = [100] number of training epochs over which a lack of performance improvement will
%                 result in termination of training. Can be specified as a single number for all
%                 layers, or one per layer.
% cfg.pyEpochs  = [1] number of epochs to run on Keras before returning 1 epoch in Matlab. Epochs in
%                 Keras are faster, but validation scores in Matlab will not be updated. Can be
%                 specified as a single number for all layers or one per layer.
% cfg.maxEpoch  = [1000] maximum number of training epochs. Can be specified as a single number for
%                 all layers or one per layer.
% cfg.minEpoch  = [0] minimum number of epochs before best network is selected and epoch restarting
%                 with cfg.minVal takes effect. Can be specified as a single number for all layers,
%                 or one per layer.
% cfg.checkEpoch= [5] number of epochs after wich validation score will be reported. Can be
%                 specified as a single number for all layers or one per layer.
% cfg.nTries    = [1] number of training attempts per layer. Network with best validation result
%                 will be chosen. Can be specified as a single number for all layers or one per
%                 layer.
% cfg.minVal    = [-Inf] minimum score that must be obtained by cfg.minEpoch. Otherwise, network
%                 will be rejected and the epoch will be restarted. Training run will not count
%                 against cfg.nTries. Can be specified as a single number for all layers or one per
%                 layer.
% cfg.valType   = ['negloss']/'score'/'area' method used to determine validation score reported at
%                 cfg.checkEpoch and used by cfg.minVal.
% cfg.keepType  = ['negloss']/'area'/'score' method used to determine which network to select.
% cfg.checkCurve= ['trainEnd']/'checkEpoch'/'no' if classification curve is not increasing, network
%                 will be discarded. 'trainEnd' only evaluates at the end of training the layer. 
%                 'checkEpoch' will check every cfg.checkEpoch for epochs > cfg.minEpoch.
%                 The training run will not count against cfg.nTries.
% cfg.truncate  = ['no'] truncates network at previous layer if additional layer doesn't improve
%                 performance as measured by cfg.keepType.
% cfg.gpu       = ['no']/'yes' whether to use the GPU for RBM pretraining. Theano's use of the GPU
%                 determined by its setup. Note that if both DeeBNet (RBM) and Theano (DNN) are set
%                 to use the GPU, a resource conflict might arise.
% cfg.theanoL   = ['c|py_nogc'] Theano linker.
%                 See http://deeplearning.net/software/theano/tutorial/modes.html#linkers for
%                 details.
%
% train_x       = training input data with dimensions nFeatures x nTrials
% train_y       = training class data using one-hot encoding, with dimensions nClass x nTrials.
% val_x         = validation input data.
% val_y         = validation class data.
%
% OUTPUTS:
% finalnet      = trained neural network.
% valscore      = validation score of neural network as measured with method in cfg.keepType.
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZ
% Author(s): Daniel D.E. Wong

assert(isfield(cfg,'layers'),'Missing field: cfg.layers');
if ~isfield(cfg,'activation')
    cfg.activation = cell(1,length(cfg.layers));
    for ii = 1:length(cfg.layers); cfg.activation{ii} = 'sigmoid'; end;
end
if ~isfield(cfg,'l1'); cfg.l1 = 0; end;
if ~isfield(cfg,'l2'); cfg.l2 = 0; end;
if ~isfield(cfg,'rbmEpochs'); cfg.rbmEpochs = 0; end;
if ~isfield(cfg,'rbmType');
    cfg.rbmType = cell(1,length(cfg.layers));
    for ii = 1:length(cfg.layers)-1; cfg.rbmType{ii} = 'generative'; end;
    cfg.rbmType{length(cfg.layers)} = 'discriminative';
end
if ~isfield(cfg,'dropout'); cfg.dropout = 0; end;
if ~isfield(cfg,'trainfcn'); cfg.trainfcn = 'sgd'; end;
if ~isfield(cfg,'freeze'); cfg.freeze = 'no'; end;
assert(isfield(cfg,'nBatch'),'Missing field: cfg.nBatch');
if ~isfield(cfg,'gwn'); cfg.gwn = 0; end;
if ~isfield(cfg,'nTerm'); cfg.nTerm = 100; end;
if ~isfield(cfg,'pyEpochs'); cfg.pyEpochs = 1; end;
if ~isfield(cfg,'maxEpoch'); cfg.maxEpoch = 1000; end;
if ~isfield(cfg,'minEpoch'); cfg.minEpoch = 0; end;
if ~isfield(cfg,'checkEpoch'); cfg.checkEpoch = 5; end;
if ~isfield(cfg,'nTries'); cfg.nTries = 1; end;
if ~isfield(cfg,'minVal'); cfg.minVal = -Inf; end;
if ~isfield(cfg,'valType'); cfg.valType = 'negloss'; end;
if ~isfield(cfg,'keepType'); cfg.keepType = 'negloss'; end;
if ~isfield(cfg,'checkCurve'); cfg.checkCurve = 'trainEnd'; end;
if ~isfield(cfg,'truncate'); cfg.truncate = 'no'; end;
if ~isfield(cfg,'gpu'); cfg.gpu = 'no'; end;
if ~isfield(cfg,'theanoL'); cfg.theanoL = 'c|py_nogc'; end;

% Convert single number to one-per-layer
for ii = {'l1','l2','rbmEpochs','dropout','nTerm','pyEpochs','maxEpoch','minEpoch', ...
        'checkEpoch','nTries','minVal'}
    if length(cfg.(ii{1}))==1; cfg.(ii{1})=repmat(cfg.(ii{1}),1,length(cfg.layers)); end;
    assert(length(cfg.(ii{1}))==length(cfg.layers),['cfg.' ii{1} ' does not have the same' ...
        ' length as cfg.layers.']);
end

% Asserts
assert(length(cfg.rbmType)==length(cfg.layers), ...
    'cfg.rbmType does not have the same length as cfg.layers.');
assert(all(cfg.minEpoch < cfg.maxEpoch), 'cfg.minEpoch should be less than cfg.maxEpoch');
assert(all(cfg.checkEpoch < cfg.maxEpoch), 'cfg.checkEpoch should be less than cfg.maxEpoch');

if any(cfg.rbmEpochs > 0)
    if ~exist('DBN','file')
        % Add DeeBNet toolbox: http://ceit.aut.ac.ir/~keyvanrad/DeeBNet%20Toolbox.html
        disp('Adding DeeBNet 3.1 to Matlab path for RBM capability.');
        addpath(fullfile(fileparts(which('co_defaults')), 'external', 'DeeBNetV3.1','DeeBNet'));
        addpath(fullfile(fileparts(which('co_defaults')), 'external', 'DeeBNetV3.1','DeepLearnToolboxGPU'));
    end
end
checkPyModule('theano');    % Generally installed with Keras, but check since we set theano.config
checkPyModule('keras');

% Set Theano linker - default cvm crashes Matlab on development system
theanocfg = py.theano.config;
theanocfg.linker = cfg.theanoL;
theanocfg.floatX = 'float32';   % For GPU compatibility - in case of difference from env settings
if strcmp(char(theanocfg.device),'gpu') && strcmp(cfg.gpu,'yes')
    warning('Both DeeBNet (RBM) and Theano (DNN) have been set to use the GPU. This might create a conflict.');
end

% Split training classes to ensure RBM minibatches are balanced
classix_train = cell(1,size(train_y,1)); minclasssz = Inf;
for ii = 1:size(train_y,1)
    classix_train{ii} = find(train_y(ii,:));
    minclasssz = min(minclasssz,length(classix_train{ii}));
end
if mod(minclasssz,cfg.nBatch) ~= 0;
    minclasssz = floor(minclasssz/cfg.nBatch)*cfg.nBatch;
    warning(['setting minclasssz=' num2str(minclasssz) ' for balancing minibatches (RBM).']);
end

% Split validation classes to balance validation score
classix_val = cell(1,size(train_y,1));
for ii = 1:size(val_y,1)
    classix_val{ii} = find(val_y(ii,:));
end

% Compute sample weights for NN training/validation
classsz = sum(train_y,2);
classw = 1 - classsz./sum(classsz);
samplew_train = zeros(size(train_y,2),1);
for ii = 1:length(classix_train); samplew_train(classix_train{ii}) = classw(ii); end;
samplew_train = samplew_train./sum(samplew_train);
classsz = sum(val_y,2);
classw = 1 - classsz./sum(classsz);
samplew_val = zeros(size(val_y,2),1);
for ii = 1:length(classix_val); samplew_val(classix_val{ii}) = classw(ii); end;
samplew_val = samplew_val./sum(samplew_val);

dropout_bool = logical(cfg.dropout);
net_p = cell(1,length(cfg.layers)); % Net from previous iteration
net_score = -Inf*ones(1,length(cfg.layers));
for ll = 1:length(cfg.layers)
    fprintf('[*] Training layer #%i\n', ll);
    
    chosenNet = [];
    chosenScore = -Inf;
    
    nTries = 0;
    while nTries < cfg.nTries(ll)
        % Create network to be trained along with training / validation features
        net = py.keras.models.Sequential();
        for ii = 1:ll
            if strcmp(cfg.freeze,'yes') && ii < ll
                trainable = py.False;   % Freeze already trained layers
            else
                trainable = py.True;
            end
            
            if cfg.l1(ii) ~= 0 && cfg.l2(ii) ~= 0
                regularizer = {'W_regularizer',py.keras.regularizers.l1l2(cfg.l1,cfg.l2)};
            elseif cfg.l1(ii) ~= 0
                regularizer = {'W_regularizer',py.keras.regularizers.l1(cfg.l1)};
            elseif cfg.l2(ii) ~= 0
                regularizer = {'W_regularizer',py.keras.regularizers.l2(cfg.l2)};
            else
                regularizer = {};
            end
            
            if ii == 1
                opt = pyargs('input_dim',py.int(size(train_x,1)), ...
                    'output_dim',py.int(cfg.layers(ii)), ...
                    'activation',cfg.activation{ii},'trainable',trainable,regularizer{:});
            else
                opt = pyargs('output_dim',py.int(cfg.layers(ii)),'activation',cfg.activation{ii},...
                    'trainable',trainable,regularizer{:});
            end
            net.add(py.keras.layers.core.Dense(opt));
            if dropout_bool(ii)
                net.add(py.keras.layers.core.Dropout(pyargs('p',py.numpy.float32(cfg.dropout(ii)))));
            end
        end
        net.add(py.keras.layers.core.Dense(pyargs(...
            'output_dim',py.int(size(train_y,1)),'activation','softmax')));
        
        
        % Copy parameters from previous iteration
        for ii = 1:ll-1
            w = py.eval('net.layers[int(i)].get_weights()', ...
                py.dict(pyargs('net',net_p{ll-1},'i',(ii-1) + sum(dropout_bool(1:ii-1)))));
            w = py.copy.deepcopy(w);
            py.eval('net.layers[int(i)].set_weights(w)', ...
                py.dict(pyargs('net',net,'i',(ii-1) + sum(dropout_bool(1:ii-1)),'w',w)));
        end

        if cfg.rbmEpochs(ll) > 0
            if ll > 1
                % Create network for computing input to new layer (omit dropout layers)
                net_tmp = py.keras.models.Sequential();
                for ii = 1:ll-1
                    if ii == 1
                        opt = pyargs('input_dim',py.int(size(train_x,1)), ...
                            'output_dim',py.int(cfg.layers(ii)), 'activation',cfg.activation{ii});
                    else
                        opt = pyargs('output_dim',py.int(cfg.layers(ii)), ...
                            'activation',cfg.activation{ii});
                    end
                    net_tmp.add(py.keras.layers.core.Dense(opt));
                    
                    % Copy weights
                    w = py.eval('net.layers[int(i)].get_weights()',py.dict(pyargs( ...
                        'net',net,'i',ii-1 + sum(dropout_bool(1:ii-1)))));
                    w = py.copy.deepcopy(w);
                    py.eval('net.layers[int(i)].set_weights(w)',py.dict(pyargs( ...
                        'net',net_tmp,'i',ii-1,'w',w)));
                end
                py.eval('net.compile(optimizer=''sgd'',loss=''categorical_crossentropy'')', ...
                    py.dict(pyargs('net',net_tmp)));
                train_feat = co_numpy2mat(net_tmp.predict(co_mat2numpy(train_x',1)))';
                val_feat = co_numpy2mat(net_tmp.predict(co_mat2numpy(val_x',1)))';
                
            else
                train_feat = train_x;
                val_feat = val_x;
            end
            dbn = dbnPreTrain(cfg.rbmEpochs(ll), cfg.rbmType{ll}, cfg.layers(ll), cfg.nBatch, ...
                cfg.gpu, train_feat', train_y', val_feat', val_y');

            % Copy DBN to final hidden layer and ouput layer
            for ii = ll:ll+1
                w = py.list();
                w.append(co_mat2numpy(dbn.W{ii-ll+1}',1));
                w.append(py.numpy.array(dbn.b{ii-ll+1}', pyargs('dtype','float32')));
                py.eval('net.layers[int(i)].set_weights(w)',py.dict(pyargs( ...
                    'net',net,'i',ii-1 + sum(dropout_bool(1:ii-1)),'w',w)));
            end

        end
        
        % Compile network
        if strcmp(cfg.trainfcn,'sgd')
            opt = pyargs('momentum',py.numpy.float32(0.9),'decay',py.numpy.float32(0.7), ...
                'nesterov',py.True);
            optimizer = py.keras.optimizers.SGD(opt);
        else
            optimizer = cfg.trainfcn;
        end
        py.eval('net.compile(optimizer=optim,loss=''categorical_crossentropy'')',py.dict(pyargs( ...
            'net',net,'optim',optimizer)));
        
        if nTries == 0; chosenNet = py.copy.deepcopy(net); end;
        
        fprintf('\tAttempt #%i.%i\n',ll,nTries+1);
        bestnet = py.copy.deepcopy(net);
        bestvalscore = -Inf; samevalscore = 0;
        
        for xx = 1:cfg.maxEpoch(ll)
            n = floor(minclasssz*size(train_y,1)/cfg.nBatch); % Batch size
            py.eval('net.fit(X,Y,nb_epoch=int(epoch),batch_size=int(batchSize),sample_weight=samplew,verbose=0)', ...
                py.dict(pyargs('net',net','X',co_mat2numpy(train_x',1),'Y',co_mat2numpy(train_y',1), ...
                'epoch',cfg.pyEpochs(ll),'batchSize',n, ...
                'samplew',py.numpy.array(samplew_train',pyargs('dtype','float32')))));

            % Validate
            if strcmp(cfg.valType,'area')
                [pcttrials,score] = makeCurve(net,val_x,val_y);
                valscore = areaundercurve(pcttrials,score);
            elseif strcmp(cfg.valType,'score')
                y = co_numpy2mat(net.predict(co_mat2numpy(val_x',1)))';
                [~,y] = max(y,[],1);    % Convert y to class indexes
                valscore = 0;
                for ii = 1:length(classix_val)
                    valscore = valscore + length(find(y(classix_val{ii})==ii))/ ...
                        length(classix_val{ii});
                end
                valscore = valscore/length(classix_val);
            else % negloss
                valscore = -double(net.evaluate(co_mat2numpy(val_x'),co_mat2numpy(val_y'), ...
                    pyargs('sample_weight',py.numpy.array(samplew_val',pyargs('dtype','float32')), ...
                    'verbose',py.int(0))));
            end

            copyFlag = false;   % Whether valscore copied to bestvalscore
            if xx >= cfg.minEpoch(ll) && valscore > bestvalscore
                copyFlag = true;
                bestvalscore = valscore;
                bestnet = py.copy.deepcopy(net);
            end
            if mod(xx,cfg.checkEpoch(ll))==0
                fprintf('\t\tIteration %i, validation score: %f, best: %f\n', ...
                    xx,valscore,bestvalscore);
                
                % Check if curve is increasing as number of classified trials decreases
                if xx >= cfg.minEpoch(ll) && strcmp(cfg.checkCurve,'checkEpoch')
                    [pcttrials,score] = makeCurve(bestnet,val_x,val_y);
                    C = corrcoef(pcttrials,score);
                    if C(1,2) >= 0; break; end; % Curve not increasing
                end
            end
            % Restart if we had a bad start
            if xx >= cfg.minEpoch(ll) && bestvalscore < cfg.minVal(ll); break; end;

            % Termination conditions
            if valscore <= bestvalscore && ~copyFlag
                samevalscore = samevalscore+1;
            else
                samevalscore = 0;
            end
            if samevalscore >= cfg.nTerm(ll); break; end;
        end
        
        % Check if validation curve is increasing as number of classified trials decreases
        if strcmp(cfg.checkCurve,'trainEnd') || strcmp(cfg.checkCurve,'checkEpoch')
            [pcttrials,score] = makeCurve(bestnet,val_x,val_y);
            C = corrcoef(pcttrials,score);
            if C(1,2) >= 0      % Curve not increasing - restart (don't increase in nTries)
                fprintf('\t\tValidation curve not increasing.\n');
                continue;
            end
        end
        
        if bestvalscore >= cfg.minVal(ll)
            if strcmp(cfg.keepType,'score') && ~strcmp(cfg.valType,'score')
                y = bestnet(val_x);
                [~,y] = max(y,[],1);    % Convert y to class indexes
                bestvalscore = 0;
                for ii = 1:length(classix_val)
                    bestvalscore = bestvalscore + length(find(y(classix_val{ii})==ii))/ ...
                        length(classix_val{ii});
                end
                bestvalscore = bestvalscore/length(classix_val);
                fprintf('\t\tScore: %f\n',bestvalscore);
            elseif strcmp(cfg.keepType,'area') && ~strcmp(cfg.valType,'area')
                [pcttrials,score] = makeCurve(bestnet,val_x,val_y);
                bestvalscore = areaundercurve(pcttrials,score);
                fprintf('\t\tArea under curve: %f\n',bestvalscore);
            elseif strcmp(cfg.keepType,'negloss') && ~strcmp(cfg.valType,'negloss')
                bestvalscore = -double(bestnet.evaluate( ...
                    co_mat2numpy(val_x'),co_mat2numpy(val_y'),pyargs( ...
                    'sample_weight',py.numpy.array(samplew_val',pyargs('dtype','float32')), ...
                    'verbose',py.int(0))));
            end
            
            if bestvalscore >= chosenScore
                chosenScore = bestvalscore;
                chosenNet = py.copy.deepcopy(bestnet);
            end
            nTries = nTries + 1;
        else
            fprintf('\t\tBest validation score does not exceed cfg.minVal. Restarting layer.\n');
        end
    end
    
    net_p{ll} = chosenNet;
    net_score(ll) = chosenScore;
    fprintf('\tChosen net score: %f\n', chosenScore);
    
    if strcmp(cfg.truncate,'yes') && ll > 1 && net_score(ll) <= net_score(ll-1); break; end;
end

if strcmp(cfg.truncate,'yes')
    [~,bestLayer] = max(net_score);
else
    bestLayer = length(cfg.layers);
end

finalnet = net_p{bestLayer};
if strcmp(cfg.freeze,'yes')
    % Reset trainable to True
    nLayers = py.eval('len(net.layers)',py.dict(pyargs('net',finalnet)));
    for ii = 1:nLayers
        layer = py.eval('net.layers[i]',py.dict(pyargs('net',finalnet,'i',ii-1)));
        layer.trainable = py.True;
    end
end
valscore = net_score(bestLayer);
fprintf('*** Final Validation Score: %f ***\n',valscore);



function [pcttrials,score] = makeCurve(net,val_x,val_y)

valclassix = cell(1,size(val_y,1));
for ii = 1:size(val_y,1)
    valclassix{ii} = find(val_y(ii,:));
end

y = co_numpy2mat(net.predict(co_mat2numpy(val_x',1)))';
[y_max,y_ix] = max(y,[],1);
score = zeros(length(y_max),1);
pcttrials = zeros(length(y_max),1);
for ii = 1:length(y_max)
    y_sel = find(y_max>=y_max(ii));
    for jj = 1:size(val_y,1)
        ix = intersect(y_sel,valclassix{jj});
        score(ii) = score(ii) + length(find(y_ix(ix)==jj))/length(ix);
    end
    score(ii) = score(ii)/size(val_y,1);

    pcttrials(ii) = length(y_sel)/length(y);
end

% Remove NaNs
nanix = isnan(score);
score = score(~nanix);
pcttrials = pcttrials(~nanix);

[pcttrials,ix] = sort(pcttrials);
score = score(ix);



function A=areaundercurve(pcttrials,score)
% Computes area under curve measure. Assumes pcttrials already sorted in ascending order and NaNs
% have been removed.

A = 0;
for ii = 1:length(pcttrials)-1
    A = A + 0.5*(score(ii)+score(ii+1))*(pcttrials(ii+1)-pcttrials(ii));
end



function dbn = dbnPreTrain(epochs,rbmType,layerSize,nBatch,gpu,train_x,train_y,val_x,val_y)
% Pretrains an RBM layer and converts it to a Matlab neural network structure

% Generate DeeBNet data structure with balanced classes    
rbmdata=DataClasses.DataStore();
rbmdata.valueType=ValueType.gaussian;
minclasssz = min(sum(train_y,1));
classix = cell(1,size(train_y,2));
for ii = 1:length(classix); classix{ii} = find(train_y(:,ii)); end;
[~,trainLabels] = max(train_y,[],2); trainLabels = trainLabels(:)-1;
rbmdata.trainData = [];
rbmdata.trainLabels = [];
for ii = 1:length(classix)
    permix = classix{ii}(randperm(length(classix{ii}),minclasssz));
    rbmdata.trainData = [rbmdata.trainData; train_x(permix,:)];
    rbmdata.trainLabels = [rbmdata.trainLabels; trainLabels(permix)];
end

minclasssz = min(sum(val_y,1));
classix = cell(1,size(val_y,2));
for ii = 1:length(classix); classix{ii} = find(val_y(:,ii)); end;
[~,valLabels] = max(val_y,[],2); valLabels = valLabels(:)-1;
rbmdata.validationData = [];
rbmdata.validationLabels = [];
for ii = 1:length(classix)
    permix = classix{ii}(randperm(length(classix{ii}),minclasssz));
    rbmdata.validationData = [rbmdata.validationData; val_x(permix,:)];
    rbmdata.validationLabels = [rbmdata.validationLabels; valLabels(permix)];
end
rbmdata.shuffle();

batchSize = floor(size(rbmdata.trainData,1)/nBatch);

% Create RBM layer
dbn = DBN('classifier');
rbmParams=RbmParameters(layerSize,ValueType.binary);
if strcmp(gpu,'yes'); rbmParams.gpu=1; else rbmParams.gpu = 0; end;
rbmParams.maxEpoch=epochs;
rbmParams.batchSize=batchSize;
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.PCD;
if strcmp(rbmType,'discriminative')
    rbmParams.rbmType=RbmType.discriminative;
    rbmParams.performanceMethod='classification';
end
dbn.addRBM(rbmParams);

% Train and convert
dbn.train(rbmdata);
dbn = DBN2ClassifierNN(dbn,rbmdata);



function net=DBN2ClassifierNN(obj,data) % Borrowed from DeeBNet toolbox code
% Converting a classifier DBN to a NN MATLAB object
%data: A DataStore class object for using its properties
output_size=max(data.trainLabels)+1;
[sizes, hidd_type] = process_rbms(obj,output_size);
net = nnsetup(sizes);
switch hidd_type
    case ValueType.binary
        net.activation_function = 'sigm';                        
end

for i = 1:length(obj.rbms)
    rbm = obj.rbms{i};
    net.W{i} = rbm.rbmParams.weight(rbm.rbmParams.numberOfVisibleSoftmax+1:end,:)';
    net.b{i} = rbm.rbmParams.hidBias';
end
rbm = obj.rbms{end};
if (rbm.rbmParams.rbmType == RbmType.discriminative)
    net.W{end} = rbm.rbmParams.weight(1:rbm.rbmParams.numberOfVisibleSoftmax,:);
    net.b{end} = rbm.rbmParams.visBias(1:rbm.rbmParams.numberOfVisibleSoftmax)';
else
    %Set last layer randomly because the DBN has not
    %discriminative RBM
    net.W{end}=0.1*randn(size(net.W{end}));
end

% Handle gpuArray transfer
for i = 1:length(net.W)
    if isa(net.W{i},'gpuArray'); net.W{i} = gather(net.W{i}); end;
    if isa(net.b{i},'gpuArray'); net.b{i} = gather(net.b{i}); end;
end


function net = nnsetup(sizes)
net.W = cell(1,length(sizes)-1);
net.b = cell(1,length(sizes)-1);
for ii = 1:length(net.W)
    net.W{ii} = zeros(sizes(ii+1),sizes(ii));
    net.b{ii} = zeros(sizes(ii+1),1);
end


%Implemented with Khademian 
function [sizes, hidd_type] = process_rbms(obj,output_size)
sizes = zeros(1, length(obj.rbms) + 1);

for i = 1:length(obj.rbms)
    sizes(i) = obj.rbms{i}.rbmParams.numHid;
end            

if (obj.rbms{end}.rbmParams.rbmType == RbmType.discriminative)
    sizes(end) = obj.rbms{end}.rbmParams.numberOfVisibleSoftmax;
else
    sizes(end) = output_size;
end

hidd_type = obj.rbms{1}.rbmParams.hiddenValueType;
for i = 2:(length(obj.rbms) - 1)
    if (obj.rbms{i}.rbmParams.hiddenValueType ~= hidd_type)
        error('DBN2DeepLearnNN: not supported - all hidden layers must have same activation function');
    end
end
sizes = [obj.rbms{1}.rbmParams.numVis-obj.rbms{1}.rbmParams.numberOfVisibleSoftmax, sizes];


function checkPyModule(module)
try py.imp.find_module(module);
catch; error(['The Python ' module ' module was not found.']); end;
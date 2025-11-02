function [finalnet,valscore,cfg] = train_layerwise(cfg,train_x,train_y,val_x,val_y)
% TRAIN_LAYERWISE trains a DNN layer by layer using minibatch training. Restricted Boltzmann
% Machine pretraining is available via the DeeBNet toolbox. The DNN itself uses the Matlab Neural
% Network toolbox. Class balancing for training and validation sets is automatically performed.
%
% INPUTS:
% cfg.layers    = array of layer sizes.
% cfg.activation= (optional) cell array indicating activation type of each layer (e.g. 'logsig'/
%                 'poslin'/other Matlab Neural Network Toolbox transfer function). Default is to use
%                 logsig for all layers.
% cfg.rbmEpochs = [0] number of epochs of RBM pretraining for each added layer. 0 indicates no
%                 RBM pretraining. Can be specified as a single number for all layers, or one per
%                 layer.
% cfg.rbmType   = (optional) cell array indicating type of RBM for each layer (e.g. {'generative',
%                 'discriminative'} for a 2 layer network. Default is for a generative network with
%                 a discriminative final hidden layer.
% cfg.trainfcn  = ['trainscg']/'traingdx' training algorithm to use.
% cfg.freeze =    ['no']/'yes'/'twopass' whether to prevent previously trained layers from being
%                 trained along with newly appended layer. The 'twopass' option performs a training
%                 pass with the previously trained layers frozen, and then with them unfrozen.
% cfg.nBatch    = number of batches.
% cfg.gwn       = [0] standard deviation of Gaussian white noise added to training inputs.
% cfg.nTerm     = [100] number of training epochs over which a lack of performance improvement will
%                 result in termination of training. Can be specified as a single number for all
%                 layers, or one per layer.
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
%                  will be rejected and the epoch will be restarted. Training run will not count
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
% cfg.gpu       = ['no']/'yes' whether to use the GPU.
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
    for ii = 1:length(cfg.layers); cfg.activation{ii} = 'logsig'; end;
end
if ~isfield(cfg,'rbmEpochs'); cfg.rbmEpochs = 0; end;
if ~isfield(cfg,'rbmType');
    cfg.rbmType = cell(1,length(cfg.layers));
    for ii = 1:length(cfg.layers)-1; cfg.rbmType{ii} = 'generative'; end;
    cfg.rbmType{length(cfg.layers)} = 'discriminative';
end
if ~isfield(cfg,'trainfcn'); cfg.trainfcn = 'trainscg'; end;
if ~isfield(cfg,'freeze'); cfg.freeze = 'no'; end;
assert(isfield(cfg,'nBatch'),'Missing field: cfg.nBatch');
if ~isfield(cfg,'gwn'); cfg.gwn = 0; end;
if ~isfield(cfg,'nTerm'); cfg.nTerm = 100; end;
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

% Convert single number to one-per-layer
for ii = {'rbmEpochs','nTerm','maxEpoch','minEpoch','checkEpoch','nTries','minVal'}
    if length(cfg.(ii{1}))==1; cfg.(ii{1})=repmat(cfg.(ii{1}),1,length(cfg.layers)); end;
    assert(length(cfg.(ii{1}))==length(cfg.layers),['cfg.' ii{1} ' does not have the same' ...
        ' length as cfg.layers.']);
end

% Asserts
assert(length(cfg.rbmType)==length(cfg.layers), ...
    'cfg.rbmType does not have the same length as cfg.layers.');
assert(all(cfg.minEpoch < cfg.maxEpoch), 'cfg.minEpoch should be less than cfg.maxEpoch');
assert(all(cfg.checkEpoch < cfg.maxEpoch), 'cfg.checkEpoch should be less than cfg.maxEpoch');

assert(exist('feedforwardnet','file')==2, 'The Matlab Neural Network Toolbox is needed.');

if any(cfg.rbmEpochs > 0)
    if ~exist('DBN','file')
        % Add DeeBNet toolbox: http://ceit.aut.ac.ir/~keyvanrad/DeeBNet%20Toolbox.html
        disp('Adding DeeBNet 3.1 to Matlab path for RBM capability.');
        addpath(fullfile(fileparts(which('co_defaults')), 'external', 'DeeBNetV3.1','DeeBNet'));
        addpath(fullfile(fileparts(which('co_defaults')), 'external', 'DeeBNetV3.1','DeepLearnToolboxGPU'));
    end
    assert(exist('DBN','file')==2, ...
        ['Please add DeeBNet package ' ...
        '(http://ceit.aut.ac.ir/~keyvanrad/DeeBNet%20Toolbox.html) to the ' ...
        'Matlab path.']);   % Cannot publicly distribute due to lack of license
end

% Split training classes to ensure minibatches are balanced - for RBM pretraining
classix_train = cell(1,size(train_y,1)); minclasssz = Inf;
for ii = 1:size(train_y,1)
    classix_train{ii} = find(train_y(ii,:));
    minclasssz = min(minclasssz,length(classix_train{ii}));
end
if mod(minclasssz,cfg.nBatch) ~= 0;
    minclasssz = floor(minclasssz/cfg.nBatch)*cfg.nBatch;
    warning(['setting minclasssz=' num2str(minclasssz) ' for balancing minibatches.']);
end

% Split validation classes to balance validation score
classix_val = cell(1,size(train_y,1));
for ii = 1:size(val_y,1)
    classix_val{ii} = find(val_y(ii,:));
end

% Compute sample weights for NN training/validation
classsz = sum(train_y,2);
classw = 1 - classsz./sum(classsz);
samplew_train = zeros(1,size(train_y,2));
for ii = 1:length(classix_train); samplew_train(classix_train{ii}) = classw(ii); end;
samplew_train = samplew_train./sum(samplew_train);
classsz = sum(val_y,2);
classw = 1 - classsz./sum(classsz);
samplew_val = zeros(1,size(val_y,2));
for ii = 1:length(classix_val); samplew_val(classix_val{ii}) = classw(ii); end;
samplew_val = samplew_val./sum(samplew_val);


net_p = cell(1,length(cfg.layers)); % Net from previous iteration
net_score = -Inf*ones(1,length(cfg.layers));
for ll = 1:length(cfg.layers)
    fprintf('[*] Training layer #%i\n', ll);
    
    chosenNet = [];
    chosenScore = -Inf;
    
    nTries = 0;
    while nTries < cfg.nTries(ll)
        % Create network to be trained along with training / validation features
        net = feedforwardnet(cfg.layers(1:ll));
        net.layers{end}.transferFcn = 'softmax';
        net.performFcn = 'crossentropy';
        net.inputs{1}.processFcns = {};
        net.outputs{end}.processFcns = {};
        net.trainParam.epochs = 1;
        net.divideFcn = 'dividetrain';
        net.trainFcn = cfg.trainfcn;
        for mm = 1:ll
            net.layers{mm}.transferFcn = cfg.activation{mm};
        end
        net = configure(net,train_x,train_y);
        net = init(net);

        if ll > 1
            % Copy parameters from previous iteration
            net.IW{1,1} = net_p{ll-1}.IW{1,1};
            net.b{1} = net_p{ll-1}.b{1};
            for mm = 1:ll-2
                net.LW{mm+1,mm} = net_p{ll-1}.LW{mm+1,mm};
                net.b{mm+1} = net_p{ll-1}.b{mm+1};
            end
        end
        if cfg.rbmEpochs(ll) > 0
            if ll > 1
                % Create network for computing input to new layer
                net_tmp = network;
                net_tmp.numInputs = 1;
                net_tmp.numLayers = ll-1; % Hidden + output layer
                net_tmp.inputConnect(1,1) = 1;
                net_tmp.outputConnect(1,ll-1) = 1;
                net_tmp.biasConnect = ones(ll-1,1);
                net_tmp.inputs{1}.size = size(train_x,1);
                for mm = 1:ll-1
                    net_tmp.layers{mm}.size = net_p{ll-1}.layers{mm}.size;
                    net_tmp.layers{mm}.transferFcn = cfg.activation{mm};
                end
                
                net_tmp.IW{1,1} = net_p{ll-1}.IW{1,1};
                net_tmp.b{1} = net_p{ll-1}.b{1};
                for mm = 1:ll-2
                    net_tmp.layerConnect(mm+1,mm) = 1;
                    net_tmp.LW{mm+1,mm} = net_p{ll-1}.LW{mm+1,mm};
                    net_tmp.b{mm+1} = net_p{ll-1}.b{mm+1};
                end
                
                train_feat = net_tmp(train_x);
                val_feat = net_tmp(val_x);
            else
                train_feat = train_x;
                val_feat = val_x;
            end
            dbn = dbnPreTrain(cfg.rbmEpochs(ll), cfg.rbmType{ll}, cfg.layers(ll), cfg.nBatch, ...
                cfg.gpu, train_feat', train_y', val_feat', val_y');

            % Copy DBN to final layer
            if ll == 1
                net.IW{1,1} = dbn.IW{1,1};
                net.b{1} = dbn.b{1};
            else
                net.LW{ll,ll-1} = dbn.IW{1,1};
                net.b{ll} = dbn.b{1};
            end
            net.LW{ll+1,ll} = dbn.LW{2,1};
            net.b{ll+1} = dbn.b{2};

        end
        
        if strcmp(cfg.freeze,'yes') || strcmp(cfg.freeze,'twopass')
            % Prevent previous layers from learning
            net = freezeLayers(net);
%             if ll == 2
%                 net.inputWeights{1,1}.learn = false;
%             elseif ll > 2
%                 net.layerWeights{ll-1,ll-2}.learn = false;
%             end
        end
        
        net.trainParam.showWindow=0;
        if nTries == 0; chosenNet = net; end;
        
        fprintf('\tAttempt #%i.%i\n',ll,nTries+1);
        bestnet = net;
        bestvalscore = -Inf; samevalscore = 0;
        
        for xx = 1:cfg.maxEpoch(ll)
            n = ceil(size(train_x,2)/cfg.nBatch); % Batch size
            permix = randperm(size(train_x,2));

            for ii = 1:cfg.nBatch
                ix = permix((ii-1)*n+1:min(ii*n,length(permix))); if isempty(ix); break; end;
                net = train(net,train_x(:,ix), train_y(:,ix), {}, {}, samplew_train(ix), ...
                    'useGPU', cfg.gpu);
            end

            % Validate
            if strcmp(cfg.valType,'area')
                [pcttrials,score] = makeCurve(net,val_x,val_y);
                valscore = areaundercurve(pcttrials,score);
            else
                y = net(val_x);
                if strcmp(cfg.valType,'negloss')
                    valscore = -crossentropy(net,val_y',y',samplew_val');
                else    % 'score'
                    [~,y] = max(y,[],1);    % Convert y to class indexes
                    valscore = 0;
                    for ii = 1:length(classix_val)
                        valscore = valscore + length(find(y(classix_val{ii})==ii))/ ...
                            length(classix_val{ii});
                    end
                    valscore = valscore/length(classix_val);
                end
            end
            
            copyFlag = false;   % Whether valscore copied to bestvalscore
            if xx >= cfg.minEpoch(ll) && valscore > bestvalscore
                copyFlag = true;
                bestvalscore = valscore;
                bestnet = net;
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
            if samevalscore >= cfg.nTerm(ll)
                if strcmp(cfg.freeze,'twopass') && ll > 1 && ~net.inputWeights{1,1}.learn
                    net = unfreezeLayers(bestnet);
                    samevalscore = 0;
                else
                    break;
                end
            end
        end
        
        % Check if validation curve is increasing as number of classified trials decreases
        if strcmp(cfg.checkCurve,'trainEnd') || strcmp(cfg.checkCurve,'checkEpoch')
            [pcttrials,score] = makeCurve(bestnet,val_x,val_y);
            C = corrcoef(pcttrials,score);
            if C(1,2) >= 0      % Curve not increasing - restart (don't increase in nTries)
                fprintf('\t\tValidation curve not increasing. Restarting layer.\n');
                continue;
            end
        end
        
        if bestvalscore >= cfg.minVal(ll)   % bestvalscore currently defined by cfg.valType
            % Need to redefine bestvalscore to cfg.keepType to compare with chosenScore
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
                y = bestnet(val_x);
                bestvalscore = -crossentropy(net,val_y',y',samplew_val');
                fprintf('\t\tNegative loss: %f\n',bestvalscore);
            end
            
            if bestvalscore >= chosenScore
                chosenScore = bestvalscore;
                chosenNet = bestnet;
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
% if strcmp(cfg.freeze,'yes')
%     % Reset learnFcn to default
%     finalnet.inputWeights{1,1}.learn = true; 
%     for ii = 1:bestLayer-1
%         finalnet.layerWeights{ii+1,ii}.learn = true;
%     end
% end
finalnet = unfreezeLayers(finalnet);    % Reset learnFcn to default
valscore = net_score(bestLayer);
fprintf('*** Final Validation Score: %f ***\n',valscore);



function [pcttrials,score] = makeCurve(net,val_x,val_y)

valclassix = cell(1,size(val_y,1));
for ii = 1:size(val_y,1)
    valclassix{ii} = find(val_y(ii,:));
end

y = net(val_x);
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
net=patternnet(5*ones(1,length(obj.rbms)));
net.trainFcn ='trainscg';
net.inputs{1}.processFcns={};
net.outputs{end}.processFcns={};
net.performFcn='mse';
net.divideFcn = '';
%Set NN structure
for i=1:length(obj.rbms)
    rbm=obj.rbms{i};
    if (i==1)
        net.inputs{1}.size=rbm.rbmParams.numVis-rbm.rbmParams.numberOfVisibleSoftmax;
    end
    switch rbm.rbmParams.hiddenValueType
        case ValueType.binary
            net.layers{i}.dimensions=rbm.rbmParams.numHid;
            net.layers{i}.transferFcn = 'logsig';
        case ValueType.probability
            net.layers{i}.dimensions=rbm.rbmParams.numHid;
            net.layers{i}.transferFcn = 'logsig';
        case ValueType.gaussian
            net.layers{i}.dimensions=rbm.rbmParams.numHid;
            net.layers{i}.transferFcn ='purelin';
    end
end %End for all RBMs
net.layers{end}.dimensions=max(data.trainLabels)+1;
net.layers{end}.transferFcn = 'logsig';
%Set NN weights
for i=1:length(obj.rbms)
    rbm=obj.rbms{i};
    if(i==1)
        net.IW{1,1}=double(tools.gather(rbm.rbmParams.weight(rbm.rbmParams.numberOfVisibleSoftmax+1:end,:)'));
        net.b{i}=double(tools.gather(rbm.rbmParams.hidBias'));
    else
        net.LW{i,i-1}=double(tools.gather(rbm.rbmParams.weight(rbm.rbmParams.numberOfVisibleSoftmax+1:end,:)'));
        net.b{i}=double(tools.gather(rbm.rbmParams.hidBias'));
    end
end %End for all RBMs
rbm=obj.rbms{end};
if (rbm.rbmParams.rbmType==RbmType.discriminative)
    net.LW{end,end-1}=double(tools.gather(rbm.rbmParams.weight(1:rbm.rbmParams.numberOfVisibleSoftmax,:)));
    net.b{end}=double(tools.gather(rbm.rbmParams.visBias(1:rbm.rbmParams.numberOfVisibleSoftmax)'));
else
    %Set last layer randomly because the DBN has not
    %discriminative RBM
    net.LW{end,end-1}=0.1*randn(size(net.LW{end,end-1}));
end



function net = freezeLayers(net)
% Freezes all except last layer
nlayers = size(net.LW,1);
for ll = 1:nlayers-2
    if ll==1
        net.inputWeights{1,1}.learn = false;
    else
        net.layerWeights{ll,ll-1}.learn = false;
    end
end



function net = unfreezeLayers(net)
% Unfreeze all layers
nlayers = size(net.LW,1);
for ll = 1:nlayers
    if ll==1
        net.inputWeights{1,1}.learn = true;
    else
        net.layerWeights{ll,ll-1}.learn = true;
    end
end
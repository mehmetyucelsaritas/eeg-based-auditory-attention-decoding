function [data,performance] = co_decode_classifier(cfg,data,decoder)
% CO_DECODE_CLASSIFIER performs classification on a dataset using a decoder trained using
% CO_TRAIN_CLASSIFIER.
%
% INPUTS:
% cfg.FIELD.cells   = ['all'] an array listing the cells from which features are taken. The user is
%                     responsible for ensuring feature labels are in the same order as in the
%                     training set.
% cfg.FIELD.test    = ['all'] test indexes. If not empty, these indexes will be used to compute
%                     class accuracy, area under curve, and confusion matrix performance measures
%                     (provided the input data contains class labels, as specified in
%                     CO_TRAIN_CLASSIFIER).
%
% data              = data structure containing features for classification.
%
% decoder           = decoder structure trained with CO_TRAIN_MATCLASSIFIER.
%
%
% OUTPUTS:
% data              = data structure with classification labels stored in the dimension label for
%                     the dimension specified by decoder.dim.
%
% performance           = performance information with the following fields:
%   .FIELDS.classNames  = array of class labels.
%   .FIELDS.d           = discriminant function output consisting of a n_samples x n_classes array.
%   .FIELDS.acccurve    = test data accuracy curve (provided input data has class labels). The
%                         discrimination score of classifications is thresholded and the accuracy is
%                         calculated at each threshold level. The first columns in .score
%                         indicate the accuracy for each class. The last column indicates the
%                         overall accuracy (with equal class weighting). The pcttrials variable
%                         indicates the number of trials classified (exceed the threshold). The area
%                         variable indicates the area under the pcttrials x score curve.
%   .FIELDS.confusion   = test data confusion matrix (provided input data has class labels). Rows
%                         correspond to actual classes, and columns correspond to predicted classes.
%                         The confusion matrix is computed for varying classification thresholds.
%                         This produces the third dimension as well as the pcttrials variable. The
%                         erasure variable indicates the fraction of trials that were not
%                         classified for each class, at each classification threshold.
%   .FIELDS.roc         = test data ROC curve (provided input data has class labels). Computes the
%                         false positive vs true positive rate with equal class weighting. First
%                         columns indicate the ROC per class, and the final column indicates the
%                         overall ROC.
%   .FIELDS.itr         = information transfer rate in bits per trial (each timepoint is considered
%                         a trial). Divide by time per trial to obtain bits per time period. The
%                         Wolpaw definition assumes the classifier must make a decision for each
%                         trial. The Nykopp definition assumes an asynchronous BCI. A bit rate is
%                         computed for different thresholds, and the Nykopp bit rate is typically
%                         taken to be the maximum. It is assumed that each class has the same
%                         probability of occuring.
%
% See also: CO_TRAIN_CLASSIFIER
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZ
% Author(s): Daniel D.E. Wong

dim = co_checkdata(cfg,data);
fields = fieldnames(cfg);

assert(strcmp(decoder.type,'class'),'Incompatible decoder.type.');

performance = [];
for ii = 1:length(fields)
    if ~isfield(cfg.(fields{ii}),'cells'); cfg.(fields{ii}).cells = 'all'; end;
    if ~isfield(cfg.(fields{ii}),'test'); cfg.(fields{ii}).test = 'all'; end;
    
    if ischar(cfg.(fields{ii}).cells) && strcmp(cfg.(fields{ii}).cells,'all')
        cells = 1:length(data.(fields{ii}));
    else
        cells = cfg.(fields{ii}).cells;
    end
    
    if ~isfield(cfg.(fields{ii}),'itr') || ~isfield(cfg.(fields{ii}).itr,'T')
        cfg.(fields{ii}).itr.T = [];
    end
    
    % Convert data to classification features
    cfgtmp = [];
    cfgtmp.(fields{ii}).dim = decoder.dim;
    cfgtmp.(fields{ii}).cells = cells;
    feat_x = co_data2class(cfgtmp, data);
    
    assert(decoder.spec.nfeat == size(feat_x,2), ...
        'Number of features does not match training data.');
    
    % Scale features
    feat_x = (feat_x-repmat(decoder.spec.scale.mean,size(feat_x,1),1))./...
        repmat(decoder.spec.scale.std,size(feat_x,1),1);
    
    switch(decoder.spec.classalgo)
        case 'svm'
            assert(exist('fitcecoc','file')==2, ...
                'The Matlab Statistics and Machine Learning Toolbox is needed for SVM.');
            performance.(fields{ii}).classNames = decoder.decoder.ClassNames;
            [feat_y,nl] = predict(decoder.decoder,feat_x);
            performance.(fields{ii}).d = 1-nl./repmat(sum(nl,2),1,size(nl,2));
        case {'nb','knn'}
            assert(exist('fitcnb','file')==2, ...
                'The Matlab Statistics and Machine Learning Toolbox is needed for Naive Bayes.');
            performance.(fields{ii}).classNames = decoder.decoder.ClassNames;
            [feat_y,performance.(fields{ii}).d] = predict(decoder.decoder,feat_x);
        case 'rf'
            assert(exist('classRF_predict','file')==2, ...
                ['Please add RandomForest-Matlab package ' ...
                '(https://github.com/jrderuiter/randomforest-matlab) RF_Class_C folder to the ' ...
                'Matlab path and compile it.']);
            performance.(fields{ii}).classNames = decoder.spec.onehotmap;
            [feat_y,votes] = classRF_predict(feat_x,decoder.decoder);
            performance.(fields{ii}).d = votes; %./repmat(sum(votes,2),1,size(votes,2));
        case 'dnn'
            assert(exist('feedforwardnet','file')==2, 'The Matlab Neural Network Toolbox is needed.');
            performance.(fields{ii}).classNames = decoder.spec.onehotmap;
            performance.(fields{ii}).d = decoder.decoder(feat_x')';
            [~,feat_y] = max(performance.(fields{ii}).d,[],2);
        case {'pysvm','pysvmlin'}
            checkPyModule('sklearn');
            feat_x = co_mat2numpy(feat_x);
            feat_y = co_numpy2mat(decoder.decoder.predict(feat_x));
            performance.(fields{ii}).classNames = co_numpy2mat(py.eval('svm.classes_', ...
                py.dict(pyargs('svm',decoder.decoder))));
            performance.(fields{ii}).d = co_numpy2mat(decoder.decoder.decision_function(feat_x));
            if size(performance.(fields{ii}).d,2) > length(performance.(fields{ii}).classNames)
                % One-vs-one
                d = zeros(size(performance.(fields{ii}).d,1),length(performance.(fields{ii}).classNames));
                c = nchoosek(1:length(performance.(fields{ii}).classNames),2);
                for jj = 1:size(c,1)
                    d(:,c(jj,1)) = d(:,c(jj(1))) + performance.(fields{ii}).d(:,jj);
                    d(:,c(jj,2)) = d(:,c(jj(1))) - performance.(fields{ii}).d(:,jj);
                end
                performance.(fields{ii}).d = d;
            elseif size(performance.(fields{ii}).d,2) == 1  % Only 2 classes
                performance.(fields{ii}).d = repmat(performance.(fields{ii}).d,1,2);
                performance.(fields{ii}).d(:,1) = -performance.(fields{ii}).d(:,1);
            end
        case {'pylogit','pynb','pyknn','pyrf'}
            checkPyModule('sklearn');
            feat_x = co_mat2numpy(feat_x);
            feat_y = co_numpy2mat(decoder.decoder.predict(feat_x));
            performance.(fields{ii}).classNames = co_numpy2mat(py.eval('decoder.classes_', ...
                py.dict(pyargs('decoder',decoder.decoder))));
            performance.(fields{ii}).d = co_numpy2mat(decoder.decoder.predict_proba(feat_x));
        case 'pydnn'
            performance.(fields{ii}).classNames = decoder.spec.onehotmap;
            performance.(fields{ii}).d = co_numpy2mat(decoder.decoder.predict(co_mat2numpy(feat_x)));
            [~,feat_y] = max(performance.(fields{ii}).d,[],2);
        case 'otherwise'
            error('Unrecognized classification algorithm.');
    end
    
    if isfield(decoder.spec,'onehotmap')
        % For RF and DNN, feat_y is an index in decoder.onehotmap - need to convert to class label
        feat_y_tmp = cell(1,size(feat_y,1));
        for jj = 1:size(feat_y,1)
            if iscell(decoder.spec.onehotmap)
                feat_y_tmp{jj} = decoder.spec.onehotmap{feat_y(jj)};
            else
                feat_y_tmp{jj} = decoder.spec.onehotmap(feat_y(jj));
            end
        end
        feat_y = feat_y_tmp;
    end
    
    % Convert feat_y to cell if necessary
    if ~iscell(feat_y)
        feat_y = mat2cell(feat_y,ones(1,length(feat_y)))';
    end
    
    % Backup old dimension labels for performance measurement
    if isfield(data.dim,decoder.dim) && isfield(data.dim.(decoder.dim),fields{ii}) ...
            && length(data.dim.(decoder.dim).(fields{ii})) >= cells(1) ...
            && ~isempty(data.dim.(decoder.dim).(fields{ii}){cells(1)})
        feat_y_old = data.dim.(decoder.dim).(fields{ii}){cells(1)};
    end
    
    % Apply feat_y to dimension labels
    for jj = cells
        data.dim.(decoder.dim).(fields{ii}){jj} = feat_y;
    end
    
    if strcmp(cfg.(fields{ii}).test,'all')
        test_ix = 1:length(feat_y);
    else
        test_ix = cfg.(fields{ii}).test;
    end
    if ~isempty(test_ix) && exist('feat_y_old','var')   % Compute more performance measures
        % Convert original to one-hot
        cfgtmp = [];
        cfgtmp.(fields{ii}).dim = decoder.dim;
        cfgtmp.(fields{ii}).cell = cells(1);
        cfgtmp.(fields{ii}).labels = performance.(fields{ii}).classNames;
        if ~iscell(cfgtmp.(fields{ii}).labels);
            cfgtmp.(fields{ii}).labels = mat2cell(cfgtmp.(fields{ii}).labels,...
                ones(1,length(cfgtmp.(fields{ii}).labels)));
        end
        datatmp = data;
        datatmp.dim.(decoder.dim).(fields{ii}){cells(1)} = feat_y_old;
        feat_y_old_vec = co_label2onehot(cfgtmp,datatmp); feat_y_old_vec = feat_y_old_vec(test_ix,:);
        d = performance.(fields{ii}).d(test_ix,:);
        testclassix = cell(1,size(d,2));
        for jj = 1:length(testclassix); testclassix{jj} = find(feat_y_old_vec(:,jj)); end;
        
        % Accuracy curve and confusion matrix
        [y_max,y_ix] = max(d,[],2);
        score = zeros(length(testclassix)+1,length(y_max));     % NClasses x N
        pcttrials = zeros(length(testclassix)+1,length(y_max));
        conf_mat = zeros(length(testclassix),length(testclassix),length(y_max));
        eras_mat = zeros(length(testclassix),length(y_max));    % Erasure channel for Nykopp ITR
        for jj = 1:length(y_max)
            y_sel = find(y_max>=y_max(jj));
            for kk = 1:length(testclassix)
                ix = intersect(y_sel,testclassix{kk});
                eras_mat(kk,jj) = length(ix);
                score(kk,jj) = length(find(y_ix(ix)==kk))/length(ix);
                score(end,jj) = score(end,jj) + score(kk,jj);
                
                conf_mat(kk,kk,jj) = length(find(y_ix(ix)==kk))/length(ix);
                for mm = 1:length(testclassix)
                    if kk==mm; continue; end;
                    conf_mat(kk,mm,jj) = length(find(y_ix(ix)==mm))/length(ix);
                end
            end
            score(end,jj) = score(end,jj)/length(testclassix);
            pcttrials(:,jj) = length(y_sel)/length(y_ix);
        end
        conf_pcttrials = pcttrials(1,:);
        
        % Class weights for ROC
        classw = zeros(1,length(test_ix));
        for jj = 1:length(testclassix); classw(testclassix{jj}) = length(testclassix{jj}); end;
        classw = 1./classw; classw = classw./sum(classw);
        
        % ROC
        true_pos = ones(length(testclassix)+1,numel(d)+1); % NClasses x N
        false_pos = ones(length(testclassix)+1,numel(d)+1);
        [~,ix] = sort(d(:));
        for jj = 1:length(ix)
            true_pos(:,jj+1) = true_pos(:,jj);
            false_pos(:,jj+1) = false_pos(:,jj);
            
            [r,c] = ind2sub(size(d),ix(jj));
            if feat_y_old_vec(r,c)
                true_pos(c,jj+1) = true_pos(c,jj+1) - classw(r)/sum(classw(testclassix{c}));
                true_pos(end,jj+1) = true_pos(end,jj+1) - classw(r);
            else
                false_pos(:,jj+1) = false_pos(:,jj+1) - classw(r)/(length(testclassix)-1);
            end
        end
        true_pos = fliplr(true_pos); false_pos = fliplr(false_pos); % Already sorted
        
        % Accuracy curve - Ignore NaNs
        nanix = isnan(score(end,:)) | isnan(pcttrials(end,:));
        score = score(:,~nanix); pcttrials = pcttrials(:,~nanix);
        % Accuracy curve - Sort by pcttrials
        for jj = 1:size(score,1)
            [score(jj,:),ix] = sort(score(jj,:)); pcttrials(jj,:) = pcttrials(jj,ix);
            [pcttrials(jj,:),ix] = sort(pcttrials(jj,:)); score(jj,:) = score(jj,ix);
        end
        if pcttrials(1,1) ~= 0
            pcttrials = [zeros(size(pcttrials,1),1), pcttrials]; score = [score(:,1), score];
        end
        if pcttrials(1,end) ~= 1
            pcttrials = [pcttrials, ones(size(pcttrials,1),1)];
            score = [score, score(:,end)];
        end
        performance.(fields{ii}).acccurve.pcttrials = pcttrials';
        performance.(fields{ii}).acccurve.score = score';
        performance.(fields{ii}).acccurve.area = zeros(1,size(pcttrials,1));
        for jj = 1:size(pcttrials,1)
            performance.(fields{ii}).acccurve.area(jj) = areaundercurve(pcttrials(jj,:),score(jj,:));
        end
        
        % Confusion matrix
        nanix = isnan(conf_pcttrials);
        conf_mat = conf_mat(:,:,~nanix); conf_pcttrials = conf_pcttrials(~nanix);
        eras_mat = eras_mat(:,~nanix);
        [conf_pcttrials,ix] = sort(conf_pcttrials); conf_mat = conf_mat(:,:,ix);
        eras_mat = eras_mat(:,ix);
        eras_mat = 1 - eras_mat./repmat(eras_mat(:,end),1,length(y_max));
        performance.(fields{ii}).confusion.confusion = conf_mat;
        performance.(fields{ii}).confusion.erasure = eras_mat;
        performance.(fields{ii}).confusion.pcttrials = conf_pcttrials;
        
        % ROC
        if false_pos(1,1) ~= 0
            false_pos = [zeros(size(false_pos,1),1) false_pos];
            true_pos = [zeros(size(true_pos,1),1), true_pos];
        end % false_pos(end) and true_pos(end) is always 1
        performance.(fields{ii}).roc.false_pos = false_pos';
        performance.(fields{ii}).roc.true_pos = true_pos';
        performance.(fields{ii}).roc.pcttrials = 1:-1/(size(false_pos,2)-1):0;
        performance.(fields{ii}).roc.pcttrials = repmat(performance.(fields{ii}).roc.pcttrials', ...
            1,size(false_pos,1));
        performance.(fields{ii}).roc.area = zeros(1,size(true_pos,1));
        for jj = 1:size(true_pos,1)
            performance.(fields{ii}).roc.area(jj) = areaundercurve(false_pos(jj,:),true_pos(jj,:));
        end
        
        % Information transfer rate - Wolpaw
        P = performance.(fields{ii}).acccurve.score(end,end);
        N = length(performance.(fields{ii}).classNames);
        performance.(fields{ii}).itr.wolpaw = log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1));  % Bits per symbol
        
        % Information transfer rate - Nykopp
        performance.(fields{ii}).itr.nykopp = zeros(1,length(performance.(fields{ii}).confusion.pcttrials));
        conferas_mat = zeros([size(conf_mat,1),size(conf_mat,2)+1,size(conf_mat,3)]);   % Confusion matrix with erasure channel
        for jj = 1:size(conf_mat,1)
            conferas_mat(jj,1:size(conf_mat,2),:) = shiftdim(conf_mat(jj,:,:),1) ...
                .*(1-repmat(eras_mat(jj,:),size(conf_mat,2),1));
            conferas_mat(jj,end,:) = eras_mat(jj,:);
        end
        for jj = 1:size(conferas_mat,3)
            Hww = conferas_mat(:,:,jj) .* log2(conferas_mat(:,:,jj)); Hww(isnan(Hww)) = 0;
            Hww = -sum(Hww(:))./N;   % Assume same probability per symbol
            p_what = sum(conferas_mat(:,:,jj),1); p_what = p_what/sum(p_what);
            Hw = p_what.*log2(p_what); Hw(isnan(Hw)) = 0; Hw = -sum(Hw);
            performance.(fields{ii}).itr.nykopp(jj) = Hw-Hww;
        end
    end
end

function checkPyModule(module)
try py.imp.find_module(module);
catch; error(['The Python ' module ' module was not found.']); end;


function A=areaundercurve(x,y)
A = 0;
for ii = 1:length(x)-1
    A = A + 0.5*(y(ii)+y(ii+1))*(x(ii+1)-x(ii));
end
% CODECODER provides methods for the saving and loading of decoders. Importantly, these methods deal
% with the serialization of Python objects.
%
%
% Copyright 2015, H2020 COCOHA Project, ENS/CNRS, DTU, Oticon, UCL, UZ
% Author(s): Daniel D.E. Wong
classdef CoDecoder<handle

    properties (Access = public)
        type;       % 'regression' or 'class'
        dim;        % Dimension info
        spec;       % Specifications of decoder
        decoder;    % The decoder
        cfg;
    end
    
    methods (Access = public)
        function obj=CoDecoder(obj_in)  % Constructor
            if nargin == 1
                if ischar(obj_in)
                    % Load file
                    obj.load(obj_in);
                elseif isa(obj_in,'CoDecoder')
                    obj.type = obj_in.type;
                    obj.dim = obj_in.dim;
                    obj.spec = obj_in.spec;
                    if strcmp(obj.type,'class')
                        if strcmp(obj.spec.classalgo,'dnn')
                            obj.decoder = clone_classnet(obj_in.decoder);
                        else
                            warning('Decoder copy operation is shallow.');
                            obj.decoder = obj_in.decoder;
                        end
                    else
                        warning('Decoder copy operation is shallow.');
                        obj.decoder = obj_in.decoder;
                    end
                    obj.cfg = obj_in.cfg;
                else
                    error('Invalid data type for obj_in.');
                end
            elseif nargin ~= 0
                error('Invalid number of arguments.');
            end
        end
        
        % Saves to decoder to file, taking care of serialization for Python objects
        function save(obj,file)
            decoder = CoDecoder(obj);
            if strcmp(obj.type,'class')
                if strncmp(obj.spec.classalgo,'py',2)
                    % Python algorithm
                    if strcmp(obj.spec.classalgo,'pydnn')
                        % keras classifier - export layer types and weights to Matlab
                        obj.checkPyModule('yaml');
                        net = [];
                        net.config = char(obj.decoder.to_yaml());
                        nLayers = py.eval('len(net.layers)',py.dict(pyargs('net',obj.decoder)));
                        net.class = cell(1,nLayers);
                        net.W = cell(1,nLayers);
                        net.b = cell(1,nLayers);
                        for ii = 1:nLayers
                            layer = py.eval('net.layers[int(i)]',py.dict(pyargs( ...
                                'net',obj.decoder,'i',ii-1)));
                            net.class{ii} = char(py.eval('layer.__class__.__name__', ...
                                py.dict(pyargs('layer',layer))));
                            
                            if strcmp(net.class{ii},'Dense')
                                w = layer.get_weights();
                                net.W{ii} = co_numpy2mat(py.eval('w[0]',py.dict(pyargs('w',w))));
                                net.b{ii} = co_numpy2mat(py.eval('w[1]',py.dict(pyargs('w',w))));
                            elseif strcmp(net.class{ii},'Dropout')
                                net.W{ii} = double(layer.p);
                            end
                        end
                        decoder.decoder = net;
                    else
                        % sklearn classifier - serializable with pickle
                        obj.checkPyModule('cPickle');
                        decoder.decoder = char(py.cPickle.dumps( ...
                            obj.decoder,py.int(0)));
                    end
                    save(file,'decoder','-v7.3');
                else
                    % Matlab algorithm - save directly
                    save(file,'decoder','-v7.3');
                end
            else
                % Regression decoders are currently all Matlab structures. Can save directly.
                save(file,'decoder','-v7.3');
            end
        end
        
        % Loads decoder from file
        function load(obj,file)
            load(file);
            obj.type = decoder.type;
            obj.dim = decoder.dim;
            obj.spec = decoder.spec;
            obj.cfg = decoder.cfg;
            
            if strcmp(decoder.type,'class') && strncmp(decoder.spec.classalgo,'py',2)
                if strcmp(decoder.spec.classalgo,'pydnn')
                    obj.checkPyModule('keras'); obj.checkPyModule('yaml');
                    net = py.keras.models.model_from_yaml(decoder.decoder.config);
                    for ii = 1:length(decoder.decoder.W)
                        w = py.list();
                        w.append(co_mat2numpy(decoder.decoder.W{ii},1));
                        w.append(py.numpy.array(decoder.decoder.b{ii}',pyargs('dtype','float32')));
                        py.eval('net.layers[int(i)].set_weights(w)',py.dict(pyargs( ...
                            'net',net,'i',ii-1,'w',w)));
                    end
                    obj.decoder = net;
                else
                    obj.checkPyModule('cPickle');
                    obj.decoder = py.cPickle.loads(decoder.decoder);
                end
            else
                % Matlab structure - load directly
                obj.decoder = decoder.decoder;
            end
        end
    end
    
    methods (Access = private)
        function checkPyModule(obj, module)
            try py.imp.find_module(module);
            catch; error(['The Python ' module ' module was not found.']); end;
        end
    end
end

function net2 = clone_classnet(net)
% Performs deep copy of a Matlab classification neural network created by toolbox
    layers = zeros(1,length(net.layers)-1); for ii = 1:length(layers); layers(ii) = net.layers{ii}.dimensions; end;
    net2 = feedforwardnet(layers);
    net2.performFcn = net.performFcn;
    net2.inputs{1}.processFcns = net.inputs{1}.processFcns;
    net2.outputs{end}.processFcns = net.outputs{end}.processFcns;
    net2.trainParam.epochs = net.trainParam.epochs;
    net2.trainFcn = net.trainFcn;
    net2.divideFcn = net.divideFcn;
    
    net2 = configure(net2,net.IW{1}',net.LW{end,end-1});
    net2 = init(net2);
    net2.IW{1} = net.IW{1};
    for ii = 1:length(net.layers)
        net2.layers{ii}.transferFcn = net.layers{ii}.transferFcn;
        if ii > 1; net2.LW{ii,ii-1} = net.LW{ii,ii-1}; end;
        net2.b{ii} = net.b{ii};
    end
end
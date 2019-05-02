classdef bpNN < handle
    % define the variable name of bpNN class
    properties
        layers      % define the nodes number of each layer in neural network
        eta         % define the learning rate
        alpha       % define the fractional gradient order
        mini_batch  % define the batch size of trainData
        weights     % weights parameters
        A           % input data of each layer
        Z           % muptiply of input data and weights parameters
        L           % the number of layers
        Delta       % transmit factors
    end
    
    properties(Constant)
        do_L2Regu = false;  % Flag of L2 regularization
        add_bias = false;   % Flag of adding bias
        lambda = 2e-5;      % steps length of L2 part
    end
    % define functions of bpNN class
    methods
        % weights parameters initialization
        function init_weights(obj)
            randn('seed',3);             % fix the random seed
            obj.L = length(obj.layers);  % compute the length of layers
            % if bias is true, add bias node
            if obj.add_bias, obj.layers(1) = obj.layers(1) + 1; end
            for ii = 1:obj.L-1
                w{ii} = randn(obj.layers(ii+1),obj.layers(ii));
            end
            obj.weights = w;
        end
        % forward broadcast
        function forward(obj, data)
            obj.L = length(obj.layers);
            a{1} = data;
            % if bias is true, add ones vector into input data
            if obj.add_bias, a{1} = [a{1}; ones(1, obj.mini_batch)]; end
            for ii = 1:obj.L-1
                [a{ii+1}, z{ii+1}]= fcn(obj.weights{ii},a{ii});
            end
            obj.A = a;
            obj.Z = z;
        end
        % back propogation
        function backward(obj, y)
            delta{obj.L} = (obj.A{obj.L} - y) .* obj.A{obj.L} .* (1-obj.A{obj.L});
            for ii = obj.L-1:-1:2
                delta{ii} = bcn(obj.weights{ii},obj.Z{ii},delta{ii+1});
            end
            obj.Delta = delta;
        end
        % gradient descent process
        function grad_dec(obj)
            for ii = 1:obj.L-1
                gw = obj.Delta{ii+1} * obj.A{ii}' / obj.mini_batch; % batch gradient descent
                gw = gw .* abs(obj.weights{ii}).^(1-obj.alpha) / gamma(2-obj.alpha);
                if obj.do_L2Regu                                    % whether do L2 regularization or not
                    gw = gw + obj.lambda * abs(obj.weights{ii}).^(1-obj.alpha) / gamma(3-obj.alpha).*sign(obj.weights{ii});
%                     fprintf('Do L2 regularization\n');
                end
                obj.weights{ii} = obj.weights{ii} - obj.eta * gw;   % weights update
            end
        end
    end
end


function delta = bcn(w, z, delta_next) % backward broadcast
    % define the activation function
    f = @(s) 1 ./ (1 + exp(-s));
    % define the derivative of activation function
    df = @(s) f(s) .* (1 - f(s));

    % backward computing (either component or vector form)
    temp = w' * delta_next;
    delta = temp .* df(z);
end

function [a_next, z_next] = fcn(w1, a1) % forward broadcast
    z_next = w1 * a1;
    a_next = 1 ./ (1 + exp(-z_next));
end

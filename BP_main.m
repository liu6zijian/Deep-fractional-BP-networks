
    % prepare the data set
    load mnist_small_matlab.mat
    % choose parameters
    train_size = size(trainLabels,2);
    test_size = size(testLabels,2);
    X_train = reshape(trainData,[],train_size);
    X_test = reshape(testData,[],test_size);
    layers = [size(X_train,1); 64; 64; 64; 10]; 
    
%     display partial images of MNIST dataset
%     X = [];
%     temp = [];
%     for i=1:10
%         for j=1:10
%             temp = [temp trainData(:,:,i*10+j)];
%         end
%         X = [X;temp];
%         temp = [];
%     end
%     imshow(X);
    
    Bp = bpNN();
    Bp.layers = layers;
    Bp.mini_batch = 200;
    is_train = input('Train (1) or test demo (0)?\n','s');
    if strcmp(is_train,'1')
        fprintf('Train process\n');
        
%     for v = 0.3:0.1:1.9
    v = input('Set a fractional order (0,2)\n');
    Bp.alpha = v;
    
    if Bp.do_L2Regu
        fprintf('Do L2 regularization\n');
    else
        fprintf('Do not L2 regularization\n');
    end
    if Bp.add_bias
        fprintf('Add bias node\n');
    else
        fprintf('Not add bias node\n');
    end
%     disp(v);
    tic;
    Bp.init_weights();
    max_iter = 300;
    Bp.eta = 3;
    EL2 = [];
    J = [];
    Acc = [];
    for iter = 1:max_iter
        for st = 1:Bp.mini_batch:train_size - Bp.mini_batch + 1
            data = X_train(:,st:st+Bp.mini_batch-1);
            y = double(trainLabels(:,st:st+Bp.mini_batch-1));
            Bp.forward(data);      % forward computation
            Bp.backward(y);    % backward computation
            Bp.grad_dec();
            J = [J 0.5*sum((Bp.A{end}(:) - y(:)).^ 2)/Bp.mini_batch];
            [~,ind_train] = max(y);
            [~,ind_pred] = max(Bp.A{end});
            Acc = [Acc sum(ind_train == ind_pred) / Bp.mini_batch];
        end
        cost = 1/2 * sum(Bp.A{end}(:) - y(:)) ^ 2;
        EL2 = [EL2 cost];
        if mod(iter,50)== 0
            fprintf('iter = %d  cost = %.3f\n',iter,cost);
        end
    end
    toc;
    plot(linspace(1,max_iter,length(J)),J);  
%     saveas(gcf,'J.jpg');
    figure
    plot(linspace(1,max_iter,length(Acc)),Acc);  
%     saveas(gcf,'Acc.jpg');

    % train accuracy
    Bp.forward(X_train);
    [~,ind_test] = max(trainLabels);
    [~,ind_pred] = max(Bp.A{end});
    train_acc = sum(ind_test == ind_pred) / train_size;
    fprintf('Accuracy on training dataset is %.2f%%\n',train_acc*100);
    
    % test accuracy
    Bp.forward(X_test);
    [~,ind_test] = max(testLabels);
    [~,ind_pred] = max(Bp.A{end});
    test_acc = sum(ind_test == ind_pred) / test_size;
    fprintf('Accuracy on test dataset is %.2f%%\n',test_acc*100);
%     W = Bp.weights;
%     save('weights.mat','W');
%     end
    else
        temp = load('weights.mat','W');
        Bp.weights = temp.W;
        % train accuracy
        Bp.forward(X_train);
        [~,ind_test] = max(trainLabels);
        [~,ind_pred] = max(Bp.A{end});
        train_acc = sum(ind_test == ind_pred) / train_size;
        fprintf('Accuracy on training dataset is %.2f%%\n',train_acc*100);

        % test accuracy
        Bp.forward(X_test);
        [~,ind_test] = max(testLabels);
        [~,ind_pred] = max(Bp.A{end});
        test_acc = sum(ind_test == ind_pred) / test_size;
        fprintf('Accuracy on test dataset is %.2f%%\n',test_acc*100);
          
    end
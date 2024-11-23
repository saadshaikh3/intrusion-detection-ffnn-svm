clear all;
close all;

% Load the dataset
p = readtable('kddcup.data_10_percent.csv', 'ReadVariableNames', false);

% Convert categorical features to numeric
newp = [];
for i = 1:42
    varName = p{:, i};
    if iscell(varName) || isstring(varName)
        [~, ~, intCol] = unique(varName);  % Numeric labels for categorical data
    else
        intCol = varName;
    end
    newp = [newp, intCol];
end

% Prepare data
dataInputs = newp(:, 1:41);  % Data without the target column
dataTargets = newp(:, 42);   % Target column

% Convert multiclass target to binary classification
dataTargets(dataTargets == 12) = 0; % Normal connections
dataTargets(dataTargets ~= 0) = 1;  % Abnormal connections

% Data Cleaning - Remove only constant columns
initialColumnCount = size(dataInputs, 2);
constantCols = var(dataInputs) == 0; % Identify columns with zero variance (constant columns)
cleanedInputs = dataInputs(:, ~constantCols); % Keep only non-constant columns
droppedColumnIndices = find(constantCols); % Identify indices of dropped columns

finalColumnCount = size(cleanedInputs, 2); % Number of columns after dropping constant columns
fprintf('Initial column count: %d\n', initialColumnCount);
fprintf('Dropped constant columns: %d\n', numel(droppedColumnIndices));
fprintf('Remaining columns after dropping constants: %d\n', finalColumnCount);
fprintf('Dropped columns at positions: %s\n', num2str(droppedColumnIndices));


% Normalize features using min-max scaling
minVals = min(cleanedInputs);
maxVals = max(cleanedInputs);
normalizedInputs = (cleanedInputs - minVals) ./ (maxVals - minVals);


% Define neural network and cross-validation parameters
hiddenLayerSizes = [10, 20, 30];  % Example layer sizes to test
learningRates = [0.01, 0.05];  % Different learning rates to test
kFolds = 5;                   % Number of cross-validation folds


perfTotal = {};
trainingFunction = 'trainlm'; % Since trainlm showed best performance during the labs

% Cross-Validation and Hyperparameter Tuning
indices = crossvalind('Kfold', dataTargets, kFolds);
counter = 0;

tic
for hiddenLayerSize = hiddenLayerSizes
    for lr = learningRates
        foldMSE = []; foldAccuracy = []; foldF1Score = [];
        
        for k = 1:kFolds
            % Split data based on fold indices
            testIdx = (indices == k); 
            trainIdx = ~testIdx;
            trainInputs = normalizedInputs(trainIdx, :)';
            trainTargets = dataTargets(trainIdx)';
            testInputs = normalizedInputs(testIdx, :)';
            testTargets = dataTargets(testIdx)';
            
            % Initialize feedforward neural network
            net = feedforwardnet(hiddenLayerSize, trainingFunction);
            
            % Set network parameters
            net.trainParam.epochs = 50;
            net.trainParam.lr = lr;
            net.performFcn = 'mse';
            net.divideFcn = 'dividetrain'; % Only use train data
            
            % Train the network
            [net, tr] = train(net, trainInputs, trainTargets);
            
            % Predict on test data
            outputs = net(testInputs);
            predictedLabels = outputs > 0.5;
            
            % Calculate performance metrics for the fold
            mse = perform(net, testTargets, outputs);
            accuracy = sum(predictedLabels == testTargets) / numel(testTargets);
            precision = sum(predictedLabels & testTargets) / sum(predictedLabels);  % TP/TP+FP
            recall = sum(predictedLabels & testTargets) / sum(testTargets);         % TP/TP+FN
            f1Score = 2 * (precision * recall) / (precision + recall);
            
            % Collect fold metrics
            foldMSE = [foldMSE, mse];
            foldAccuracy = [foldAccuracy, accuracy];
            foldF1Score = [foldF1Score, f1Score];
        end
        
        % Average performance across folds
        avgMSE = mean(foldMSE);
        avgAccuracy = mean(foldAccuracy);
        avgF1Score = mean(foldF1Score);
        
        % Store results for this configuration to performance table
        perfTotal = [perfTotal; {'Feed Forward NN', hiddenLayerSize, lr, avgMSE, avgAccuracy, avgF1Score}];
        counter = counter + 1;
        
        % Display results for this configuration
        fprintf('Config %d - HiddenLayerSize: %d, LearningRate: %.2f\n', counter, hiddenLayerSize, lr);
        fprintf('Avg MSE: %.4f, Avg Accuracy: %.4f, Avg F1 Score: %.4f\n\n', avgMSE, avgAccuracy, avgF1Score);
    end
end
toc

tic
% SVM Model Training and Evaluation
svmFoldMSE = []; svmFoldAccuracy = []; svmFoldF1Score = [];

for k = 1:kFolds
    testIdx = (indices == k);
    trainIdx = ~testIdx;
    
    trainInputs = normalizedInputs(trainIdx, :);
    trainTargets = dataTargets(trainIdx);
    testInputs = normalizedInputs(testIdx, :);
    testTargets = dataTargets(testIdx);
    
    % Train SVM model
    svmModel = fitcsvm(trainInputs, trainTargets, 'KernelFunction', 'linear');
    
    % Predict on test data
    predictedLabels = predict(svmModel, testInputs);
    
    % Calculate performance metrics
    mse = mean((predictedLabels - testTargets) .^ 2);
    accuracy = sum(predictedLabels == testTargets) / numel(testTargets);
    precision = sum(predictedLabels & testTargets) / sum(predictedLabels);
    recall = sum(predictedLabels & testTargets) / sum(testTargets);
    f1Score = 2 * (precision * recall) / (precision + recall);
    
    % Collect fold metrics for SVM
    svmFoldMSE = [svmFoldMSE, mse];
    svmFoldAccuracy = [svmFoldAccuracy, accuracy];
    svmFoldF1Score = [svmFoldF1Score, f1Score];
end
toc
% Average SVM performance across folds
svmAvgMSE = mean(svmFoldMSE);
svmAvgAccuracy = mean(svmFoldAccuracy);
svmAvgF1Score = mean(svmFoldF1Score);

fprintf('SVM - Avg MSE: %.4f, Avg Accuracy: %.4f, Avg F1 Score: %.4f\n\n', svmAvgMSE, svmAvgAccuracy, svmAvgF1Score);

% Append SVM results to performance table
perfTotal = [perfTotal; {'SVM', NaN, NaN, svmAvgMSE, svmAvgAccuracy, svmAvgF1Score}];

% Display summary of all results in a table
resultsTable = cell2table(perfTotal, ...
    'VariableNames', {'Model','HiddenLayerSize', 'LearningRate', 'MSE', 'Accuracy', 'F1Score'});

disp('Summary of Performance for Different Parameter Configurations (Feedforward NN and SVM):');
disp(resultsTable);

% view(net)


% [coeff, score, ~] = pca(normalizedInputs);
% pc1 = score(:, 1);
% pc2 = score(:, 2);
% 
% %2d graph PCA
% figure;
% gscatter(pc1, pc2, dataTargets, 'rb', 'xo');
% xlabel('Principal Component 1');
% ylabel('Principal Component 2');
% title('PCA Visualization of Normalized Inputs');
% legend('Class 0 (Normal)', 'Class 1 (Attack)');
% grid on;
% 
% %3d graph PCA
% pc3 = score(:, 3);
% figure;
% scatter3(pc1, pc2, pc3, 15, dataTargets, 'filled');
% xlabel('Principal Component 1');
% ylabel('Principal Component 2');
% zlabel('Principal Component 3');
% title('3D PCA Visualization of Normalized Inputs');
% grid on;
% colorbar;
% view(45, 30);
data = readtable('updated_online_shoppers_intention_updated.csv');

data.Properties.VariableNames = matlab.lang.makeValidName(data.Properties.VariableNames);

% Convert categorical/string columns to numeric
if iscell(data.Month)
    data.Month = grp2idx(categorical(data.Month)); % Convert 'Month' to numeric
end

if iscell(data.VisitorType)
    data.VisitorType = grp2idx(categorical(data.VisitorType)); % Convert 'VisitorType' to numeric
end
% Separate features (X) and target (y)
X = table2array(data(:, 1:end-1)); % Assuming last column is 'Revenue'
y = table2array(data(:, end));    % Assuming 'Revenue' is the target

% Split data into training and test sets (80-20 split)
cv = cvpartition(size(X, 1), 'Holdout', 0.2);
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test = X(test(cv), :);
y_test = y(test(cv), :);

% Standardize features
mean_X = mean(X_train, 1);
std_X = std(X_train, 1);
X_train = (X_train - mean_X) ./ std_X;
X_test = (X_test - mean_X) ./ std_X;


% Train SVM with Class Weights
costMatrix = [0 1; 10 0]; % Penalize misclassification of Revenue = 1
SVMModel = fitcsvm(X_train, y_train, 'KernelFunction', 'linear', ...
                   'Standardize', true, ...
                   'ClassNames', [0, 1], ...
                   'Cost', costMatrix);

% Predict on Test Set
[predicted_labels, scores] = predict(SVMModel, X_test);

% Confusion Matrix and Metrics
confusion_mat = confusionmat(y_test, predicted_labels);
TP = confusion_mat(2, 2); % True Positives
FP = confusion_mat(1, 2); % False Positives
TN = confusion_mat(1, 1); % True Negatives
FN = confusion_mat(2, 1); % False Negatives

accuracy = (TP + TN) / sum(confusion_mat(:));
precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1_score = 2 * (precision * recall) / (precision + recall);

% Display Metrics
fprintf('Accuracy: %.4f\n', accuracy);
if ~isnan(precision)
    fprintf('Precision: %.4f\n', precision);
else
    fprintf('Precision: NaN (no positive predictions)\n');
end
fprintf('Recall: %.4f\n', recall);
if ~isnan(f1_score)
    fprintf('F1-Score: %.4f\n', f1_score);
else
    fprintf('F1-Score: NaN (undefined)\n');
end

% Confusion Matrix
disp('Confusion Matrix:');
disp(confusion_mat);

% Cross-Validation Accuracy
cv_svm = crossval(SVMModel, 'KFold', 5);
cv_accuracy = 1 - kfoldLoss(cv_svm);
fprintf('Cross-Validation Accuracy: %.4f\n', cv_accuracy);

% Plot ROC Curve
[~, scores_train] = predict(SVMModel, X_train);
[~, scores_test] = predict(SVMModel, X_test);

[X_ROC, Y_ROC, ~, AUC] = perfcurve(y_test, scores(:, 2), 1); % Assuming column 2 is the positive class
figure;
plot(X_ROC, Y_ROC, 'b-', 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC Curve (AUC = %.4f)', AUC));
grid on;


% Visualize confusion matrix
figure;
confusionchart(confusion_mat, {'No Revenue', 'Revenue'});
% Load and prepare the data
data = readtable('updated_online_shoppers_intention_updated.csv');

% Convert categorical variables (e.g., 'VisitorType', 'Month') to numeric
if iscell(data.VisitorType)
    data.VisitorType = grp2idx(data.VisitorType); % Convert 'VisitorType' to numeric
end
if iscell(data.Month)
    data.Month = grp2idx(data.Month); % Convert 'Month' to numeric
end

if iscell(data.Administrative)
    data.Administrative = grp2idx(data.Administrative); % Convert 'Administrative' to numeric
end

if iscell(data.Weekend)
    data.Weekend = grp2idx(data.Weekend); % Convert 'Weekend' to numeric
end

% Extract features and target
X = table2array(data(:, 1:end-1)); % All columns except the last
y = data.Revenue;                  % Target variable (binary)



% Perform 5-fold cross-validation
k = 5;
cv = cvpartition(height(data), 'KFold', k);

cv_accuracy = zeros(k, 1);    % Store accuracy for each fold
cv_precision = zeros(k, 1);   % Store precision for each fold
cv_recall = zeros(k, 1);      % Store recall for each fold
cv_f1 = zeros(k, 1);          % Store F1-score for each fold
auc_values = zeros(k, 1);     % Store AUC values for each fold
confusion_matrices = cell(k, 1); % Store confusion matrices for each fold

figure;
hold on;

for i = 1:k
    % Split data into training and testing for this fold
    train_idx = training(cv, i);
    test_idx = test(cv, i);
    
    X_train = X(train_idx, :);
    y_train = y(train_idx);
    X_test = X(test_idx, :);
    y_test = y(test_idx);
    
    % Apply SMOTE to handle class imbalance in the training set
    [X_train_smote, y_train_smote] = smote(X_train, y_train, 'NumNeighbors', 5);
    
    % Fit logistic regression model
    B = mnrfit(X_train_smote, y_train_smote + 1); % Add 1 to y since mnrfit expects 1-based classes
    
    % Predict probabilities and convert to class predictions
    probabilities = mnrval(B, X_test);
    predicted_labels = probabilities(:, 2) >= 0.5; % Binary classification threshold at 0.5
    
    % Ensure y_test and predicted_labels are the same type
    y_test = double(y_test); % Convert y_test to double
    predicted_labels = double(predicted_labels); % Convert predicted labels to double
    
    % Calculate confusion matrix for this fold
    confusion_matrix = confusionmat(y_test, predicted_labels);
    confusion_matrices{i} = confusion_matrix;
    
    % Extract metrics from confusion matrix
    true_negatives = confusion_matrix(1, 1);
    false_positives = confusion_matrix(1, 2);
    false_negatives = confusion_matrix(2, 1);
    true_positives = confusion_matrix(2, 2);
    
    % Calculate metrics
    accuracy = (true_positives + true_negatives) / sum(confusion_matrix(:));
    precision = true_positives / (true_positives + false_positives);
    recall = true_positives / (true_positives + false_negatives);
    f1_score = 2 * (precision * recall) / (precision + recall);
    
    % Handle cases where precision or recall might be NaN
    if isnan(precision), precision = 0; end
    if isnan(recall), recall = 0; end
    if isnan(f1_score), f1_score = 0; end
    
    % Store metrics for this fold
    cv_accuracy(i) = accuracy;
    cv_precision(i) = precision;
    cv_recall(i) = recall;
    cv_f1(i) = f1_score;
    
    % Calculate AUC for ROC curve
    [fpr, tpr, ~, auc] = perfcurve(y_test, probabilities(:, 2), 1);
    auc_values(i) = auc;
    
    % Plot ROC curve for this fold
    plot(fpr, tpr, 'DisplayName', sprintf('Fold %d (AUC = %.2f)', i, auc));
end

% Average metrics across all folds
mean_accuracy = mean(cv_accuracy);
mean_precision = mean(cv_precision);
mean_recall = mean(cv_recall);
mean_f1 = mean(cv_f1);
mean_auc = mean(auc_values);

% Combine confusion matrices across folds
combined_confusion_matrix = sum(cat(3, confusion_matrices{:}), 3);

% Display results
fprintf('Cross-Validation Results:\n');
fprintf('Accuracy: %.4f\n', mean_accuracy);
fprintf('Precision: %.4f\n', mean_precision);
fprintf('Recall: %.4f\n', mean_recall);
fprintf('F1-Score: %.4f\n', mean_f1);
fprintf('AUC: %.4f\n', mean_auc);
fprintf('\nCombined Confusion Matrix:\n');
disp(combined_confusion_matrix);

% Plot random classifier line for ROC
plot([0 1], [0 1], '--', 'DisplayName', 'Random Classifier');

% Customize ROC plot
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve for Each Fold');
legend('show');
hold off;

% Display cross-validation accuracy for each fold
fprintf('Mean Cross-Validation Accuracy: %.4f\n', mean(cv_accuracy));
for i = 1:k
    fprintf('Fold %d: %.4f\n', i, cv_accuracy(i));
end

% Visualize confusion matrix
figure;
confusionchart(combined_confusion_matrix, {'No Revenue', 'Revenue'});


function [X_resampled, y_resampled] = smote(X, y, varargin)

% SMOTE: Synthetic Minority Over-sampling Technique


% Parse input arguments

p = inputParser;

addParameter(p, 'NumNeighbors', 5, @isnumeric);

parse(p, varargin{:});

k = p.Results.NumNeighbors;



% Find minority class

classes = unique(y);

class_counts = histc(y, classes);

[~, minority_class_idx] = min(class_counts);

minority_class = classes(minority_class_idx);



% Separate minority and majority classes

X_minority = X(y == minority_class, :);

X_majority = X(y ~= minority_class, :);

y_minority = y(y == minority_class);

y_majority = y(y ~= minority_class);



% Generate synthetic samples

num_minority_samples = size(X_minority, 1);

num_synthetic_samples = size(X_majority, 1) - num_minority_samples;

synthetic_samples = zeros(num_synthetic_samples, size(X, 2));



for i = 1:num_synthetic_samples

    % Randomly select a minority sample

    idx = randi(num_minority_samples);

    sample = X_minority(idx, :);

    

    % Find k nearest neighbors

    distances = sqrt(sum((X_minority - sample).^2, 2));

    [~, neighbor_idxs] = sort(distances);

    neighbors = X_minority(neighbor_idxs(2:k+1), :);

    

    % Randomly select a neighbor and generate a synthetic sample

    neighbor = neighbors(randi(k), :);

    synthetic_sample = sample + rand * (neighbor - sample);

    synthetic_samples(i, :) = synthetic_sample;

end



% Combine original and synthetic samples

X_resampled = [X; synthetic_samples];

y_resampled = [y; repmat(minority_class, num_synthetic_samples, 1)];

end

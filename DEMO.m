% Run FAFCN on a MAT file containing X (samples by features) and Y (labels).

demo_dir = fileparts(mfilename('fullpath'));
dataset_file = fullfile(demo_dir, 'compare data', 'jaffe50_uni.mat');

measures_dir = fullfile(demo_dir, 'Measures');
if exist(measures_dir, 'dir') == 7
    addpath(measures_dir);
end

dataset = load(dataset_file, 'X', 'Y');
if ~isfield(dataset, 'X') || ~isfield(dataset, 'Y')
    error('The dataset MAT file must contain X and Y.');
end
validateattributes(dataset.Y, {'numeric', 'logical'}, ...
    {'vector', 'nonempty', 'real', 'finite'});
n = size(dataset.X, 1);
if numel(dataset.Y) ~= n
    error('X must have one row per sample and Y must have one label per row.');
end

[~, ~, true_labels] = unique(full(double(dataset.Y(:))));
c = max(true_labels);

% Use the original ORL_32 settings, capped for smaller datasets.
m = min(256, 2^floor(log2(n)));%m_list_raw       = 2.^(4:10);    16, 32, ..., 1024
 k = min(10, m - 1);
lambda = 4;% lambda_list      = [0.05, 0.1, 0.5, 1, 5, 10, 50];

rng(1, 'twister');
predicted_labels = FAFCN(dataset.X, c, m, k, lambda);

metrics = ClusteringMeasure_All(true_labels, predicted_labels);
ACC = metrics(1);
NMI = metrics(2);
ARI = RandIndex(true_labels, predicted_labels);
F1 = metrics(6);

fprintf('ACC: %.4f  NMI: %.4f  ARI: %.4f  F1: %.4f\n', ACC, NMI, ARI, F1);

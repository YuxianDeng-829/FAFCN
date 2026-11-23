function [labels, Y, Z, obj, numIter] = FAFCN(X, c, m, k, lambda)
%FADC Cluster samples with TSPT initialization and DC/CD optimization.
%   labels = FADC(X, c, m, k, lambda) clusters the rows of X into c groups.
%   m is the number of anchors (a power of two) and k is the number of
%   nearest anchors used by AGf. lambda is the nuclear-norm weight.
%   Optional outputs are sample memberships Y, anchor memberships Z,
%   objective values obj, and the number of optimization iterations numIter.

    validateattributes(X, {'numeric'}, {'2d', 'nonempty', 'real', 'finite'});
    validateattributes(c, {'numeric'}, {'scalar', 'real', 'finite', 'integer', 'positive'});
    c = double(c);
    validateattributes(m, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'integer', '>=', max(2, c), '<=', size(X, 1)});
    m = double(m);
    if mod(log2(m), 1) ~= 0
        error('FADC:InvalidAnchorCount', 'm must be a power of two for AGf.');
    end
    validateattributes(k, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'integer', 'positive', '<', m});
    validateattributes(lambda, {'numeric'}, {'scalar', 'real', 'finite', 'nonnegative'});
    k = double(k);
    lambda = double(lambda);

    opts.maxIter = 100;
    opts.tol = 1e-6;
    opts.eps = 1e-12;
    opts.innerZ = 15;
    opts.innerZTol = 1e-7;
    opts.verbose = false;
    opts.reconScale = 1.0;
    opts.monotone = true;
    opts.kmeansMaxIter = 100;

    % Build the sample-to-anchor graph and normalize it in both directions.
    [B_raw, ~] = AGf(zscore(double(X)), log2(m), k, 1);
    B = normalizeRows(B_raw, opts.eps);
    B_col = normalizeCols(B_raw, opts.eps);
    m_actual = size(B, 2);
    if size(B, 1) ~= size(X, 1) || m_actual < c
        error('FADC:InvalidAnchorGraph', ...
            'AGf must return one row per sample and at least c anchor columns.');
    end

    % Decompose the TSPT anchor graph to initialize anchor memberships Z.
    S0 = full(B_col' * B);
    S0(S0 < 0) = 0;
    S0 = normalizeRows(S0, opts.eps);
    W = (S0 + S0') / 2;
    anchor_labels = spectralClusteringLabels(W, c, opts.eps, opts.kmeansMaxIter);
    Z = zeros(m_actual, c);
    Z(sub2ind(size(Z), (1:m_actual)', anchor_labels)) = 1;

    % Propagate anchor memberships to the samples, then refine both.
    Y = projectRowsSimplex_fast(full(B * Z));
    [Y, Z, obj, numIter] = FADC_solver_DC_CD_fast(B, Y, Z, lambda, opts);
    [~, labels] = max(Y, [], 2);
end


function labels = spectralClusteringLabels(W, c, eps0, maxIter)
% Spectral clustering labels on graph W.
%
% Use normalized Laplacian:
%
%   Lsym = I - D^{-1/2} W D^{-1/2}
%
% Then cluster the smallest c eigenvectors by K-means++.

    W = double(W);
    W = (W + W') / 2;
    W(W < 0) = 0;

    m = size(W, 1);

    d = sum(W, 2);
    d(d <= eps0) = eps0;

    D_inv_sqrt = spdiags(1 ./ sqrt(d), 0, m, m);

    Lsym = speye(m) - D_inv_sqrt * sparse(W) * D_inv_sqrt;
    Lsym = (Lsym + Lsym') / 2;

    try
        if m > c + 1
            [U, ~] = eigs(Lsym, c, 'smallestreal');
        else
            [Uall, Eval] = eig(full(Lsym));
            [~, idx] = sort(diag(Eval), 'ascend');
            U = Uall(:, idx(1:c));
        end
    catch
        [Uall, Eval] = eig(full(Lsym));
        [~, idx] = sort(diag(Eval), 'ascend');
        U = Uall(:, idx(1:c));
    end

    U = real(U);
    U = U ./ (sqrt(sum(U.^2, 2)) + eps0);

    labels = simpleKMeansLabels(U, c, maxIter, eps0);
end


function labels = simpleKMeansLabels(X, c, maxIter, eps0)
% K-means with K-means++ seeding on the spectral embedding.

    X = double(X);

    [n, d] = size(X);

    if n < c
        error('Number of data points is smaller than cluster number.');
    end

    X(isnan(X)) = 0;
    X(isinf(X)) = 0;

    centers = kmeansPlusPlusCenters(X, c, eps0);

    labels = ones(n, 1);

    for iter = 1:maxIter

        old_labels = labels;

        % Assign labels
        distMat = squaredEuclideanDistance(X, centers);
        [~, labels] = min(distMat, [], 2);

        % Update centers
        for j = 1:c
            idx_j = find(labels == j);

            if isempty(idx_j)
                % Empty cluster repair: choose farthest point from existing centers
                minDist = min(distMat, [], 2);
                [~, farIdx] = max(minDist);
                centers(j, :) = X(farIdx, :);
                labels(farIdx) = j;
            else
                centers(j, :) = mean(X(idx_j, :), 1);
            end
        end

        if isequal(labels, old_labels)
            break;
        end
    end

    labels = repairEmptyLabels(labels, c);
end


function centers = kmeansPlusPlusCenters(X, c, eps0)
% K-means++ center initialization.

    [n, d] = size(X);

    centers = zeros(c, d);

    firstIdx = randi(n);
    centers(1, :) = X(firstIdx, :);

    closestDist = squaredEuclideanDistance(X, centers(1, :));

    for j = 2:c

        prob = closestDist / max(sum(closestDist), eps0);

        r = rand();
        cp = cumsum(prob);
        idx = find(cp >= r, 1, 'first');

        if isempty(idx)
            idx = randi(n);
        end

        centers(j, :) = X(idx, :);

        newDist = squaredEuclideanDistance(X, centers(j, :));
        closestDist = min(closestDist, newDist);
    end
end


function labels = repairEmptyLabels(labels, c)
    labels = labels(:);
    for j = 1:c
        if ~any(labels == j)
            counts = accumarray(labels, 1, [c, 1]);
            [~, largest_cluster] = max(counts);
            idx = find(labels == largest_cluster, 1, 'last');
            labels(idx) = j;
        end
    end
end


function D2 = squaredEuclideanDistance(A, B)
% Compute squared Euclidean distance between rows of A and rows of B.
%
% A: n*d
% B: k*d
% D2: n*k

    A2 = sum(A.^2, 2);
    B2 = sum(B.^2, 2)';

    D2 = bsxfun(@plus, A2, B2) - 2 * (A * B');
    D2 = max(D2, 0);
end


function [Y, Z, obj, converge] = FADC_solver_DC_CD_fast(B, Y, Z, lambda, opts)
% Optimize:
%
%   min_{Z1=1,Z>=0,Y1=1,Y>=0}
%       reconScale * ||Y - BZ||_F^2 - lambda * Tr(sqrt(Y'Y))
%
% Method:
%   Y-step: DC/MM closed projection
%   Z-step: incremental coordinate descent

    maxIter    = opts.maxIter;
    tol        = opts.tol;
    eps0       = opts.eps;
    verbose    = opts.verbose;
    reconScale = opts.reconScale;
    monotone   = opts.monotone;

    [~, m] = size(B);

    B = sparse(double(B));
    B = normalizeRows(B, eps0);

    H = full(B' * B);
    H = (H + H') / 2;

    obj = zeros(maxIter, 1);

    obj_prev = computeFADCObjective(B, Y, Z, lambda, reconScale, eps0);

    for iter = 1:maxIter

        Y_old = Y;
        Z_old = Z;
        obj_old = obj_prev;

        %% ===================== Step 1: update Y by DC/MM ===============
        Q = full(B * Z);

        D = nuclearNormSubgradient(Y, eps0);

        Y_new = projectRowsSimplex_fast( ...
            Q + (lambda / (2 * reconScale)) * D ...
        );

        if monotone
            objY = computeFADCObjective(B, Y_new, Z, lambda, reconScale, eps0);

            if objY > obj_old + 1e-10
                Y_new = lineSearchY(B, Y, Y_new, Z, lambda, reconScale, eps0, obj_old);
            end
        end

        Y = Y_new;


        %% ===================== Step 2: update Z by fast CD ==============
        P = full(B' * Y);
        HZ = H * Z;

        for sweep = 1:opts.innerZ

            Z_before = Z;

            order = randperm(m);

            for ii = 1:m

                h = order(ii);

                Hhh = H(h, h);

                if Hhh <= eps0
                    continue;
                end

                z_old = Z(h, :);

                s_h = HZ(h, :) - Hhh * z_old;

                v = (P(h, :) - s_h) / Hhh;

                z_new = projectSimplex_fast(v);

                delta = z_new - z_old;

                if norm(delta) > 0
                    Z(h, :) = z_new;
                    HZ = HZ + H(:, h) * delta;
                end
            end

            diffZ = norm(Z - Z_before, 'fro') / max(1, norm(Z_before, 'fro'));

            if diffZ < opts.innerZTol
                break;
            end
        end


        %% ===================== Step 3: monotone safeguard ===============
        obj_curr = computeFADCObjective(B, Y, Z, lambda, reconScale, eps0);

        if monotone && obj_curr > obj_old + 1e-10

            accepted = false;
            alpha_list = [0.5, 0.25, 0.1, 0.05, 0.01];

            for aa = 1:length(alpha_list)

                alpha = alpha_list(aa);

                Y_try = (1 - alpha) * Y_old + alpha * Y;
                Z_try = (1 - alpha) * Z_old + alpha * Z;

                obj_try = computeFADCObjective(B, Y_try, Z_try, lambda, reconScale, eps0);

                if obj_try <= obj_old + 1e-10
                    Y = Y_try;
                    Z = Z_try;
                    obj_curr = obj_try;
                    accepted = true;
                    break;
                end
            end

            if ~accepted
                Y = Y_old;
                Z = Z_old;
                obj_curr = obj_old;
            end
        end


        %% ===================== Step 4: convergence ======================
        obj(iter) = obj_curr;

        relObjChange = abs(obj(iter) - obj_old) / max(1, abs(obj_old));

        varDiff = ...
            norm(Y - Y_old, 'fro') / max(1, norm(Y_old, 'fro')) + ...
            norm(Z - Z_old, 'fro') / max(1, norm(Z_old, 'fro'));

        if verbose
            fprintf('Iter %03d: obj = %.10e, relObj = %.3e, varDiff = %.3e\n', ...
                iter, obj(iter), relObjChange, varDiff);
        end

        if relObjChange < tol || varDiff < tol
            converge = iter;
            obj = obj(1:iter);
            return;
        end

        obj_prev = obj(iter);
    end

    converge = maxIter;
    obj = obj(1:maxIter);
end


function D = nuclearNormSubgradient(Y, eps0)
% Subgradient of:
%
%   ||Y||_* = Tr(sqrt(Y'Y)).
%
% If Y = U S V', one valid subgradient is U V'.

    C = Y' * Y;
    C = (C + C') / 2;

    [V, S] = eig(C);

    sigma = sqrt(max(real(diag(S)), 0));
    maxSigma = max(sigma);

    if maxSigma <= eps0
        D = zeros(size(Y));
        return;
    end

    rankTol = max(1e-10 * maxSigma, eps0);

    active = sigma > rankTol;

    if ~any(active)
        D = zeros(size(Y));
        return;
    end

    V_r = V(:, active);
    sigma_r = sigma(active);

    D = Y * V_r * diag(1 ./ sigma_r) * V_r';
    D = real(D);
end


function val = computeFADCObjective(B, Y, Z, lambda, reconScale, eps0)
% Objective:
%
%   reconScale * ||Y-BZ||_F^2 - lambda * Tr(sqrt(Y'Y))

    BZ = full(B * Z);

    reconTerm = reconScale * sum(sum((Y - BZ).^2));

    nuclearTerm = traceSqrtYTY(Y, eps0);

    val = reconTerm - lambda * nuclearTerm;
end


function val = traceSqrtYTY(Y, eps0)
% Compute:
%
%   Tr(sqrt(Y'Y))

    C = Y' * Y;
    C = (C + C') / 2;

    eigVals = eig(C);
    eigVals = max(real(eigVals), 0);

    val = sum(sqrt(eigVals + eps0));
end


function Y_new = lineSearchY(B, Y_old, Y_new, Z, lambda, reconScale, eps0, obj_ref)
% Convex combination keeps simplex feasibility.

    alpha_list = [0.5, 0.25, 0.1, 0.05, 0.01];

    for i = 1:length(alpha_list)

        alpha = alpha_list(i);

        Y_try = (1 - alpha) * Y_old + alpha * Y_new;

        obj_try = computeFADCObjective(B, Y_try, Z, lambda, reconScale, eps0);

        if obj_try <= obj_ref + 1e-10
            Y_new = Y_try;
            return;
        end
    end

    Y_new = Y_old;
end


function Bn = normalizeRows(B, eps0)
% Row-normalization:
%
%   Bn * 1 = 1

    B = sparse(double(B));
    B(B < 0) = 0;

    [n, ~] = size(B);

    rs = full(sum(B, 2));
    rs(rs <= eps0) = 1;

    Bn = spdiags(1 ./ rs, 0, n, n) * B;

    Bn(isnan(Bn)) = 0;
    Bn(isinf(Bn)) = 0;
end


function Bn = normalizeCols(B, eps0)
% Column-normalization:
%
%   each column sums to 1.

    B = sparse(double(B));
    B(B < 0) = 0;

    [~, m] = size(B);

    cs = full(sum(B, 1));
    cs(cs <= eps0) = 1;

    Bn = B * spdiags(1 ./ cs', 0, m, m);

    Bn(isnan(Bn)) = 0;
    Bn(isinf(Bn)) = 0;
end


function X = projectRowsSimplex_fast(V)
% Project each row of V onto probability simplex:
%
%   x >= 0, sum(x) = 1

    [n, c] = size(V);
    X = zeros(n, c);

    for i = 1:n
        X(i, :) = projectSimplex_fast(V(i, :));
    end
end


function x = projectSimplex_fast(v)
% Project vector v onto probability simplex:
%
%   x >= 0, sum(x) = 1

    v = v(:);
    c = length(v);

    u = sort(v, 'descend');
    cssv = cumsum(u);

    rho = find(u + (1 - cssv) ./ (1:c)' > 0, 1, 'last');

    if isempty(rho)
        x = ones(c, 1) / c;
    else
        theta = (cssv(rho) - 1) / rho;
        x = max(v - theta, 0);
    end

    s = sum(x);

    if s <= 0
        x = ones(c, 1) / c;
    else
        x = x / s;
    end

    x = x(:)';
end

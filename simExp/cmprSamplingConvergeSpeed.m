% Compare the convergence speed of Langevin sampling with dimensions.

% Wen-Hao Zhang, Mar. 27, 2024
% UT Southwestern Medical Center
% wenhao.zhang@utsouthwestern.edu

nDimArray = 1:10;
tauL = 1; % time constant of Langevin sampling
dt = 0.01; % time step for simulation

nSteps = 1e3;
nTrials = 100;

KLDivArray = zeros(nSteps, length(nDimArray));
for iterDim = 1: length(nDimArray)
    fprintf('Progress: %d/%d\n', iterDim, length(nDimArray));
    nDim = nDimArray(iterDim);

    %% Set the parameters of posteriors
    x = zeros(nDim, 1);

    % Prior precision matrix
    Lambda_Stim = 1;
    LMat = - ones(nDim) + nDim * diag(ones(1,nDim));
    LMat = Lambda_Stim * LMat;

    % Likelihood precision matrix
    Lambda = 4*ones(nDim, 1);
    Lambda = diag(Lambda);

    Omega = Lambda + LMat; % Posterior precision
    muS = Omega \ Lambda * x;

    %%
    sMat = diag(ones(1,nDim)) - Omega * dt/ (2*tauL);
    xMat = Lambda*dt / (2*tauL);

    sArray = zeros(nDim, nTrials, nSteps);
    s_mean = zeros(nDim, nSteps);
    s_cov = zeros(nDim, nDim, nSteps);

    sArray(:,:,1) = randn(nDim, nTrials);
    s_mean(:,1) = mean(sArray(:,:,1),2);
    s_cov(:,:,1) = cov(sArray(:,:,1)');

    for t = 1: nSteps-1
        sArray(:,:,t+1) = sMat * sArray(:,:,t) + xMat*x + sqrt(dt/tauL) * randn(nDim,nTrials,1);

        sampleSet = reshape(sArray(:,:,1:t+1), nDim, []);
        s_mean(:,t+1) = mean(sampleSet, 2);
        s_cov(:,:,t+1) = cov(sampleSet');
    end
    clear sampleSet

    % sCell = mat2cell(sArray, nDim, nTrials, ones(1, nSteps));
    % sCell = squeeze(sCell);
    %
    % s_mean = cellfun(@(x) mean(x, 2), sCell, 'UniformOutput',false);
    % s_cov = cellfun(@(x) cov(x'), sCell, 'UniformOutput',false);

    s_mean = squeeze(mat2cell(s_mean, nDim, ones(1,nSteps)));
    s_cov = squeeze(mat2cell(s_cov, nDim, nDim, ones(1,nSteps)));
    s_mean = s_mean(:);
    s_cov = s_cov(:);

    Cov0 = inv(Omega);
    KLDiv = cellfun(@(sample_mean, sample_cov) KLDivFunc(x, Cov0, sample_mean, sample_cov), ...
        s_mean, s_cov);

    KLDivArray(:, iterDim) = KLDiv;

end

% Normalize the KL divergence based on the max value
KLDivArray_Norm = KLDivArray./ max(KLDivArray, [], 1);
%%

cSpec = cool(length(nDimArray));

figure
hold on
for iterDim = 1: length(nDimArray)
    plot((1:nSteps)*dt/tauL, KLDivArray_Norm(:,iterDim), 'color', cSpec(iterDim,:))
end
axis square
ylabel('KL divergence (normalized)')
xlabel('Time (/\tau_s)')
set(gca, 'ytick', 0:0.25:1)

colormap(cSpec);
clim(nDimArray([1,end]))
colorbar('Ticks', nDimArray)



%%
function KLDiv = KLDivFunc(mu0, cov0, mu1, cov1)
KLDiv = trace(cov1 \ cov0) - length(mu0) + ((mu1 - mu0)' / cov1) * (mu1 - mu0) ...
    + log(det(cov1)) - log(det(cov0));
KLDiv = KLDiv / 2;
end
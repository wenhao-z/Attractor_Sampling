% Test the simulation of an OU process
% \tau dx = -alpha x dt + \sigma \sqrt(2\tau) dW

% Wen-Hao Zhang
% Apr. 3, 2020
% University of Pittsburgh

nDim = 2;
tauL = 1:10; % time constant of Langevin sampling

dt = 0.01; % time step for simulation

x = [-1; 1];

% Prior precision matrix
Lambda_Stim = 1;
LMat = - ones(nDim) + 2 * diag(ones(1,nDim));
LMat = Lambda_Stim * LMat;

% Likelihood precision matrix
Lambda = [1;2];
Lambda = diag(Lambda);

Omega = Lambda + LMat; % Posterior precision
muS = Omega \ Lambda * x;

%%
tStat = 1e4;

meanS = zeros(nDim, length(tauL));
covS = zeros(nDim, nDim, length(tauL));


for iterPar = 1: length(tauL)
    fprintf('Progress: %d/%d\n', iterPar, length(tauL));
    
    nSteps = 5e5 * tauL(iterPar);
    sMat = diag(ones(1,nDim)) - Omega * dt./ (2*tauL(iterPar));
    xMat = Lambda*dt ./ (2*tauL(iterPar));
    
    rng(0);
    sArray = zeros(nDim, nSteps);
    for t = 1: nSteps
        sArray(:,t+1) = sMat * sArray(:,t) + xMat*x + sqrt(dt/tauL(iterPar)) * randn(nDim,1);
    end
    
    % Statistics
    meanS(:,iterPar) = mean(sArray(:, tStat+1:end), 2);
    covS(:,:,iterPar) = cov(sArray(:, tStat+1:end)');
    
end

%%
SigmaS = inv(Omega);

figure
hold on
cMap = lines(3);
% Posterior parameters
plot(tauL([1,end]), SigmaS(1,1)*ones(1,2), 'color', cMap(1,:))
plot(tauL([1,end]), SigmaS(2,2)*ones(1,2), 'color', cMap(2,:))
plot(tauL([1,end]), SigmaS(1,2)*ones(1,2), 'color', cMap(3,:))

% Empirical covariance
plot(tauL, squeeze(covS(1,1,:)), 'o', 'color', cMap(1,:))
plot(tauL, squeeze(covS(2,2,:)), 'o', 'color', cMap(2,:))
plot(tauL, squeeze(covS(1,2,:)), 'o', 'color', cMap(3,:))

legend('V(s_1)', 'V(s_2)', 'cov(s_1,s_2)')
xlabel('Sampling time constant \tau_L')
ylabel('Variance')
ylim([0, 0.8])

axis square
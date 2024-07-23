% Test the simulation of an OU process
% \tau dx = -alpha x dt + \sigma \sqrt(2\tau) dW

% Wen-Hao Zhang
% Apr. 3, 2020
% University of Pittsburgh

alpha = 1;
dt = 0.01;
nSteps = 5e5;

flagTest = 2;
switch flagTest
    case 1
        sigma = 1: 10;
        sigma = sigma(:);
        tau = 1;
        
        xArray = zeros(length(sigma), nSteps);
    case 2
        sigma = 1;
        tau = 1:10;
        tau = tau(:);
        
        xArray = zeros(length(tau), nSteps);
end

for t = 1: nSteps
    xArray(:,t+1) = (1-alpha*dt./tau).*xArray(:,t) + sigma .* sqrt(2*dt./tau) * randn(1);
    %     xArray(:,t+1) = (1-alpha*dt./tau).*xArray(:,t) + sigma * sqrt(2*dt./tau) .* randn(length(tau),1);
end


%% -------??-------??-------??-------??-------??-------??-------??-------??
% Statistics
tStat = 1e4;
varX = var(xArray(:, tStat+1:end), 0, 2);

% Cross correlation function
lenCCFunc = 1e3;

CCFunc = zeros(size(xArray,1), 2*lenCCFunc + 1);
for iterPar = 1: size(xArray,1)
    [CCFunc(iterPar,:), tLag] = xcorr(xArray(iterPar, tStat+1:end)', lenCCFunc);
end

CCFunc = CCFunc(:, (end+1)/2:end)./max(CCFunc,[],2);
tLag = tLag((end+1)/2:end) * dt;

% Fit the time constant of cross correlation function
tau_CC = zeros(1, size(xArray,1));
for iterPar = 1: size(xArray,1)
    % fitObj = fit(tLag', CCFunc(iterPar,:)', 'exp1');
    switch flagTest
        case 1
            nLen = min(5*tau/alpha/dt,lenCCFunc+1);
        case 2
            nLen = min(5*tau(iterPar)/alpha/dt,lenCCFunc+1);
    end
    fitObj = fit(tLag(1:nLen)', CCFunc(iterPar, 1:nLen)', 'exp1');
    tau_CC(iterPar) = - 1./fitObj.b;
end

%%
figure
% plot(tau, varX);
% hold on
% plot(tau([1,end]), sigma.^2/alpha*ones(1,2))

switch flagTest
    case 1
        subplot(1,2,1)
        plot(sigma, varX, 'o');
        hold on
        plot(sigma, sigma.^2/alpha)
        xlabel('sigma')
        ylabel('Var(x)')
        axis square
        
        % Auto correlation function.
        subplot(1,2,2)
        [CCFunc, lags] = xcorr(xArray(1,:), 5e2);
        lags = lags((end+1)/2:end);
        CCFunc = CCFunc((end+1)/2:end)./max(CCFunc);
        
        CCFunc_Theory = exp(-abs(lags)*alpha*dt/tau);
        
        plot(lags*dt, CCFunc)
        hold on
        plot(lags*dt, CCFunc_Theory)
        axis square
        xlabel('Time (\tau)')
        ylabel('Cross correlation')
        ylim([-2e-2, 1])
        legend('Sim', 'Theory')
        
    case 2
        subplot(1,2,1)
        
        plot(tLag, CCFunc')
        xlabel('Time lag')
        ylabel('Cross correlation')
        axis square
        
        subplot(1,2,2)
        hold on
        plot(tau, tau./alpha)
        plot(tau, tau_CC, 'o')
        axis square
        
        xlabel('Time constant in sampling')
        ylabel('Time constant (fit in CC)')
        axis([0.8 max(tau) 0.8 max(tau)])
end



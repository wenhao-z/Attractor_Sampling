% Test the simulation of an OU process
% \tau dx = -alpha x dt + \sigma \sqrt(2\tau) dW

% Wen-Hao Zhang
% Apr. 3, 2020
% University of Pittsburgh

alpha = 2; 
% sigma = 1;
sigma = 1: 10;
sigma = sigma(:);

tau = 1;
% tau = 1:0.5:5;
% tau = tau(:);
dt = 0.01;

nSteps = 5e5;
% xArray = zeros(length(tau), nSteps);
xArray = zeros(length(sigma), nSteps);

for t = 1: nSteps
    xArray(:,t+1) = (1-alpha*dt./tau).*xArray(:,t) + sigma .* sqrt(2*dt./tau) * randn(1);
%     xArray(:,t+1) = (1-alpha*dt./tau).*xArray(:,t) + sigma * sqrt(2*dt./tau) .* randn(length(tau),1);
end

varX = var(xArray(:, 1e3+1:end), 0, 2);

%% 
figure
% plot(tau, varX); 
% hold on
% plot(tau([1,end]), sigma.^2/alpha*ones(1,2))

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



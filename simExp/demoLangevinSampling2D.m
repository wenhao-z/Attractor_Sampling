% Test the simulation of an OU process
% \tau dx = -alpha x dt + \sigma \sqrt(2\tau) dW

% Wen-Hao Zhang
% Apr. 3, 2020
% University of Pittsburgh

nDim = 2;
tauL = 1; % time constant of Langevin sampling
dt = 0.01; % time step for simulation

% x = [-1; 1];
x = [0;0];

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
nSteps = 5e5;

sMat = diag(ones(1,nDim)) - Omega * dt/ (2*tauL);
xMat = Lambda*dt / (2*tauL);

sArray = zeros(nDim, nSteps);
for t = 1: nSteps
    sArray(:,t+1) = sMat * sArray(:,t) + xMat*x + sqrt(dt/tauL) * randn(nDim,1);
end


%% Statistics
tStat = 1e4;
meanS = mean(sArray(:, tStat+1:end), 2);
covS = cov(sArray(:, tStat+1:end)');

%% Plot

tPlot = 1e5+2e2;
tlenPlot = 5e2;

figure
% Plot the empirical distribution of samples
hAxe = plotJointMarginalHist(sArray(1,tStat+1:end), sArray(2,tStat+1:end));

% Get the range of coordinates
xLim = get(hAxe(1), 'xlim');
yLim = get(hAxe(1), 'ylim');
xGrid = linspace(xLim(1), xLim(end), 1e2+1);
yGrid = linspace(yLim(1), yLim(end), 1e2+1);

% Contour or the color image of the empirical distribution of samples
[X,Y] = ndgrid(xGrid, yGrid);
pdfSample = mvnpdf([X(:), Y(:)], meanS', covS);
pdfSample = reshape(pdfSample, size(X));
% imagesc(hAxe(1), xGrid, yGrid, pdfSample')
% contour(X,Y, pdfSample)
contourf(X,Y, pdfSample, 'linestyle', 'none')

% Use the colormap of the same color series
addpath(fullfile(Path_RootDir, 'plotCode'));
% cMap = getColorMapPosNegDat([0, max(pdfSample(:))], 64);
% colormap(cMap);
cMap = flipud(gray(64));
colormap(cMap)
caxis([-1e-2, max(pdfSample(:))])
axis xy

% Plot the distribution of posterior predicted by Bayes theorem
SigmaS = inv(Omega);
fPostBayes = @(x,y) ( ([x;y] - muS)' * Omega * ([x;y]-muS) - 9);
hEllipse = fimplicit(hAxe(1), fPostBayes, [muS(1) + 5*covS(1)*[-1, 1], muS(2) + 5*covS(4)*[-1, 1]], ...
    'color', 'k', 'linestyle', '--', 'linew', 2);
plot(hAxe(2), xGrid, normpdf(xGrid, muS(1), sqrt(SigmaS(1))), '--k', 'linew',2)
plot(hAxe(3), normpdf(yGrid, muS(2), sqrt(SigmaS(2,2))), yGrid, '--k', 'linew',2)


% Plot an example of trajectory
cMap = cool(tlenPlot);
for iter = 1: (tlenPlot-1)
   plot(sArray(1,tPlot+(iter:iter+1)), sArray(2,tPlot+(iter:iter+1)), 'color', cMap(iter,:)); 
end
% hPlot = plot(sArray(1,tPlot+(1:tlenPlot)), sArray(2,tPlot+(1:tlenPlot)), 'k');
% cMap = [uint8(cool(tlenPlot)*255) uint8(ones(tlenPlot,1))].';
% drawnow
% set(hPlot.Edge, 'ColorBinding','interpolated', 'ColorData', cMap)

axes(hAxe(1))
xlabel('s_1')
ylabel('s_2')

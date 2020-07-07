% Test the performance of decentralized system under different parameters

% Wen-Hao Zhang, Oct-6-2016
% wenhaoz1@andrew.cmu.edu
% @Carnegie Mellon University

setWorkPath;

% Load parameters
parsCoupledCANNs;

NetPars.AmplRatio = 1*ones(NetPars.numNets,1);
NetPars.JrcRatio = 0.5;
NetPars.JrpRatio = 0.8;
NetPars.tLen = 1e3;

NetPars.fanoFactor = 0.5;
NetPars.Posi = 2*[-1;1];

% Generate grid of parameters
[parGrid, dimPar] = paramGrid(NetPars);
parGrid = arrayfun(@(x) getDependentPars(x), parGrid);

%% Net Simulation
NetStat = struct('BumpPos', [], ...
    'meanBumpPos', [], ...
    'varBumpPos', [], ...
    'OHeightAvg', [], ...
    'OAvgXTime', [], ...
    'OStdXTime', [], ...
    'UHeightAvg', []);
NetStat = repmat(NetStat, size(parGrid));

tStart = clock;
for iterPar = 1: numel(parGrid)
    fprintf('Progress: %d/%d\n', iterPar, numel(parGrid));
    netpars = parGrid(iterPar);
    
    % Network input
    InputSet = makeNetInput([], netpars);
    
    % Run simulation
    outArgs = struct('InputSet', [], 'NetStat', NetStat(iterPar));
    [InputSet, NetStat(iterPar)] = simCoupledAttractors1(InputSet, netpars, outArgs);
end

tEnd = clock;

%% Parameters of sample distributions
PreMat_Sample = {NetStat.varBumpPos};
PreMat_Sample = cellfun(@inv, PreMat_Sample, 'uniformout', 0);
PreMat_Sample = reshape(cell2mat(PreMat_Sample), NetPars.numNets, NetPars.numNets, []);

%% Theoretical prediction

wfwd = 8/3^(3/2) * NetPars.fanoFactor * 2.5;

% New version includes the spatial convolution of the feedforward input.
PreMat_LH = 2*sqrt(pi) * diag(parGrid(1).Ampl) / NetPars.TunWidth/ wfwd;

% ------------------------------------------------------
% The variance of noise is the firing rate bump
% varBumpPosTheory = [NetStat.OHeightAvg]./ [NetStat.UHeightAvg]*NetPars.TunWidth ...
%     ./ (2*sqrt(pi) .*[parGrid.Ampl]);

% The variance of noise is the synaptic input bump
varBumpPosTheory = 4*NetPars.fanoFactor*NetPars.TunWidth/sqrt(pi)/ 3^(3/2) ./ [parGrid.Ampl];

% ------------------------------------------------------
% Prediction of mean of samples
% Find the prior precision which best explains the sample distribution
meanSampleTheory = zeros(NetPars.numNets, numel(parGrid));
Lambda_s = zeros(1, numel(parGrid));
KLD = zeros(1, numel(parGrid));

for iter = 1: numel(NetStat)
    meanSampleTheory(:,iter) = NetStat(iter).varBumpPos * ...
        PreMat_LH * NetPars.Posi;
    [Lambda_s(iter), KLD(iter)] = findPriorPrecision(NetStat(iter).meanBumpPos, NetStat(iter).varBumpPos, ...
        NetPars.Posi, PreMat_LH);
end
meanSampleTheory = reshape(meanSampleTheory, [NetPars.numNets, size(parGrid)]);
OmegaTheory = PreMat_LH + Lambda_s * (2*diag(ones(1, NetPars.numNets)) - ones(size(NetPars.numNets)));

%% Plot

tPlot = 1e4+1e2;
tlenPlot = 5e2;

figure
% Plot the empirical distribution of samples
hAxe = plotJointMarginalHist(NetStat.BumpPos(1,NetPars.tStat/NetPars.dt+1:end), ...
    NetStat.BumpPos(2,NetPars.tStat/NetPars.dt+1:end));

% Get the range of coordinates
xLim = get(hAxe(1), 'xlim');
yLim = get(hAxe(1), 'ylim');
xGrid = linspace(xLim(1), xLim(end), 1e2+1);
yGrid = linspace(yLim(1), yLim(end), 1e2+1);

% Contour or the color image of the empirical distribution of samples
[X,Y] = ndgrid(xGrid, yGrid);
pdfSample = mvnpdf([X(:), Y(:)], NetStat.meanBumpPos', NetStat.varBumpPos);
pdfSample = reshape(pdfSample, size(X));
% imagesc(hAxe(1), xGrid, yGrid, pdfSample')
contour(X,Y, pdfSample)
caxis([-5e-2, max(pdfSample(:))])
axis xy

% Plot the distribution of posterior predicted by Bayes theorem
SigmaS = inv(OmegaTheory);
fPostBayes = @(x,y) ( ([x;y] - meanSampleTheory)' * OmegaTheory * ([x;y]-muS) - 9);
hEllipse = fimplicit(hAxe(1), fPostBayes, ...
    [SigmaS(1) + 5*SigmaS(1)*[-1, 1], meanSampleTheory(2) + 5*SigmaS(4)*[-1, 1]], ...
    'color', 'k', 'linestyle', '--', 'linew', 2);
plot(hAxe(2), xGrid, normpdf(xGrid, meanSampleTheory(1), sqrt(SigmaS(1))), '--k', 'linew',2)
plot(hAxe(3), normpdf(yGrid, meanSampleTheory(2), sqrt(SigmaS(2,2))), yGrid, '--k', 'linew',2)


% Plot an example of trajectory
cMap = cool(tlenPlot);
for iter = 1: (tlenPlot-1)
    plot(NetStat.BumpPos(1,tPlot+(iter:iter+1)), NetStat.BumpPos(2,tPlot+(iter:iter+1)), 'color', cMap(iter,:));
end

axes(hAxe(1))
xlabel('s_1')
ylabel('s_2')

% Test the performance of decentralized system under different parameters

% Wen-Hao Zhang, Oct-6-2016
% wenhaoz1@andrew.cmu.edu
% @Carnegie Mellon University

setWorkPath;

% Load parameters
parsSingleCANN;

NetPars.AmplRatio = 0.02;
NetPars.JrcRatio = 0.5;
NetPars.fanoFactor = 0.5;
NetPars.tLen = 5e4 * NetPars.tau;
NetPars.seedNois = sum(clock);

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

%% Theoretical prediction

wfwd = 8/3^(3/2) * NetPars.fanoFactor * 2.5;

% Old code directly specifies the input strength
% PreMat_LH = sqrt(2*pi) * [parGrid.Ampl] .* NetPars.rho./ (sqrt(2)*NetPars.TunWidth);

% New version includes the spatial convolution of the feedforward input.
PreMat_LH = 2*sqrt(pi) * [parGrid.Ampl] / NetPars.TunWidth/ wfwd;

% ------------------------------------------------------------
% The variance of noise is the firing rate bump 
% varBumpPosTheory = [NetStat.OHeightAvg]./ [NetStat.UHeightAvg]*NetPars.TunWidth ...
%     ./ (2*sqrt(pi) .*[parGrid.Ampl]);

% The variance of noise is the synaptic input bump
varBumpPosTheory = 4*NetPars.fanoFactor*NetPars.TunWidth/sqrt(pi)/ 3^(3/2) ./ [parGrid.Ampl];

% ------------------------------------------------------------
% Auto correlation function.

[CCFunc, tLag] = xcorr(NetStat.BumpPos(1,NetPars.tStat/NetPars.dt+1:end), 1e3);
CCFunc = CCFunc((end+1)/2:end)./max(CCFunc);
tLag = tLag((end+1)/2:end) * NetPars.dt;
% CCFunc_Theory = exp(-abs(tLag)*parGrid.Ampl/NetPars.tau/NetStat.UHeightAvg);
CCFunc_Theory = exp(-abs(tLag)* parGrid.Ampl * NetPars.rho /NetPars.tau/NetStat.UHeightAvg/sqrt(2)*2.5);

%% Analyze the step size with population firing rate

nBin = 0.5*NetPars.tau/NetPars.dt; % Decoding time window
nBootStrap = 100;

Rate = squeeze(InputSet.O(:,1, 50*NetPars.tau/NetPars.dt+1 : end));
% Rate = squeeze(InputSet.O(:,1, NetPars.tStat/NetPars.dt+1 : end));
% Rate = squeeze(InputSet.O);
Rate = mat2cell(Rate, size(Rate,1), nBin*ones(1, size(Rate,2) / nBin));
Rate = cellfun(@(x) mean(x,2), Rate, 'uniformoutput', false);
Rate = cell2mat(Rate);

BumpPos = statBumpPos(Rate, NetPars);
diffPos = diff(BumpPos);

PopRate = sum(Rate(end/2-45:end/2+45,:), 1);
% PopRate = sum(Rate, 1);

[N, edges, bin] = histcounts(PopRate(1:end-1), 100);

varStepSize = zeros(1, length(N));
std_varStepSize = zeros(1, length(N));
for iter = 1 : length(N)
    diffPosTmp = diffPos(bin == iter);
    varStepSize(iter) = var(diffPosTmp);

    % Bootstrap to get the std. of statistics
    if length(diffPosTmp) <= 1
        continue
    end
    varStepSize_bootstat = bootstrp(nBootStrap, @(x) var(x), diffPosTmp);
    std_varStepSize(iter) = std(varStepSize_bootstat);
end
clear diffPosTmp

% Fit
fitMdl = fittype(@(a,b,x) a./x + b);

IdxStart = find(N>50, 1, 'first');
IdxEnd = find(N>50, 1, 'last');
fitObj = fit(edges(IdxStart:IdxEnd)', varStepSize(IdxStart:IdxEnd)', fitMdl);

%%
figure
hAxe(1) = subplot(3,1,1);
plot(PopRate(1:end-1), (diffPos), '.')
ylabel('Step size')
title(['AmplRatio:' num2str(NetPars.AmplRatio), ' JrcRatio:' num2str(NetPars.JrcRatio)]);

hAxe(2) = subplot(3,1,2);
hold on
plot(edges(1:end-1), varStepSize, '-.')
plot(edges(1:end-1), varStepSize - std_varStepSize)
plot(edges(1:end-1), varStepSize + std_varStepSize)
plot(fitObj, edges(1:end-1), varStepSize);
xlabel('Population Firing Rate')
ylabel('Var. of step size')

hAxe(3) = subplot(3,1,3);
plot(edges(1:end-1), N)
hold on
plot(edges([1, end-1]), 50*ones(1,2), '--k')

linkaxes(hAxe, 'x')
xlim(edges([IdxStart, IdxEnd]))

